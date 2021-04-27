from typing import Tuple

from thexp.nest.trainer.losses import *
from thexp.calculate.tensor import onehot
import numpy as np
from . import tricks


class MixMatchLoss(Loss):

    def sharpen_(self, x: torch.Tensor, T=0.5):
        """
        让概率分布变的更 sharp，即倾向于 onehot
        :param x: prediction, sum(x,dim=-1) = 1
        :param T: temperature, default is 0.5
        :return:
        """
        with torch.no_grad():
            temp = torch.pow(x, 1 / T)
            return temp / temp.sum(dim=1, keepdims=True)

    def label_guesses_(self, *logits):
        """根据K次增广猜测"""
        with torch.no_grad():
            k = len(logits)
            un_logits = torch.cat(logits)  # type:torch.Tensor
            targets_u = torch.softmax(un_logits, dim=1) \
                            .view(k, -1, un_logits.shape[-1]) \
                            .sum(dim=0) / k
            targets_u = targets_u.detach()
            return targets_u

    def mixup_(self,
               imgs: torch.Tensor, targets: torch.Tensor,
               beta=0.75, reids=None, target_b=None):
        """
        普通的mixup操作
        """
        if reids is not None:
            idx = reids
        else:
            idx = torch.randperm(imgs.size(0))

        input_a, input_b = imgs, imgs[idx]
        target_a = targets
        if target_b is None:
            target_b = targets[idx]
        else:
            target_b = target_b[idx]

        l = np.random.beta(beta, beta, size=imgs.shape[0])
        l = np.max([l, 1 - l], axis=0)
        l = torch.tensor(l, device=input_a.device, dtype=torch.float)
        # torch.tensordot()

        # mixed_input = l * input_a + (1 - l) * input_b
        mixed_input = tricks.elementwise_mul(l, input_a) + tricks.elementwise_mul(1 - l, input_b)
        # mixed_target = l * target_a + (1 - l) * target_b
        mixed_target = tricks.elementwise_mul(l, target_a) + tricks.elementwise_mul(1 - l, target_b)

        return mixed_input, mixed_target

    def mixmatch_up_(self,
                     sup_imgs: torch.Tensor, un_sup_imgs: List[torch.Tensor],
                     sup_targets: torch.Tensor, un_targets: torch.Tensor,
                     beta=0.75):
        """
        使用过MixMatch的方法对有标签和无标签数据进行mixup混合

        注意其中 un_sup_imgs 是一个list，包含K次增广图片batch
        而 un_targets 则只是一个 tensor，代表所有k次增广图片的标签
        """
        imgs = torch.cat((sup_imgs, *un_sup_imgs))
        targets = torch.cat([sup_targets, *[un_targets for _ in range(len(un_sup_imgs))]])
        return self.mixup_(imgs, targets, beta)

    def loss_ce_with_masked_(self,
                             logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                             meter: Meter = None, name: str = 'Lmce'):
        loss = (F.cross_entropy(logits, labels, reduction='none') * mask).mean()
        if meter is not None:
            meter[name] = loss
        return loss

    def loss_ce_with_targets_masked_(self,
                                     logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                                     meter: Meter = None, name: str = "Ltce"):
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1) * mask)
        if meter is not None:
            meter[name] = loss
        return loss


class IEGLoss(Loss):
    def logit_norm_(self, v, use_logit_norm=True):
        return v
        # if not use_logit_norm:
        #     return v
        # return v * torch.rsqrt(torch.pow(v, 2).mean() + 1e-8)

    def loss_kl_ieg_(self, q_logits: torch.Tensor, p_logits: torch.Tensor,
                     consistency_factor,
                     n_classes,
                     meter: Meter = None,
                     name='Lkl'):
        q = torch.softmax(q_logits, dim=1).detach()
        per_example_kl_loss = q * (
                torch.log_softmax(q_logits, dim=1) - torch.log_softmax(p_logits, dim=1))

        loss = per_example_kl_loss.mean() * n_classes * consistency_factor
        if meter is not None:
            meter[name] = loss
        return loss

    def create_metanet(self):  # -> Tuple[MetaModule, MetaSGD]:
        raise NotImplementedError()


class ClsContrastLoss(Loss):

    def loss_loss_sim_contrast_(self, features: torch.Tensor, anchor: torch.Tensor,
                                temperature=0.5,
                                meter: Meter = None, name: str = 'Lsim'):
        """
        选取特征中离得最近的作为正例，其余的作为负例。
        :param features:  用于区分相似度，获取mask的feature
        :param anchor:  用于最终计算的 feature，需要保留梯度
        :param temperature:
        :param meter:
        :param name:
        :return:
        """
        dot_product = torch.matmul(features, features.T) / temperature
        values, indices = dot_product.topk(2, dim=-1)

        dot_product = torch.matmul(anchor, anchor.T) / temperature
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = torch.zeros_like(dot_product).scatter(1, indices[:, 1:2], 1)
        logits_mask = torch.ones_like(dot_product).scatter(1, indices[:, 0:1], 0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        loss = torch.relu(loss)

        if meter is not None:
            meter[name] = loss

        return indices[:, 1], loss

    def loss_mt_contrast_(self, features: torch.Tensor, pys: torch.Tensor,
                          temperature=0.5,
                          meter: Meter = None, name: str = 'Lmtt'):
        """
        选择同类的做正例，类别有 pys 指定（伪标签）
        :param features: [batchsize, feature_dim]
        :param cls_mask:
        :param meter:
        :param name:
        :return:
        """
        size = pys.shape[0]
        pys_ = pys.unsqueeze(-1).repeat([1, size])
        cls_mask = (pys == pys_).float()

        dot_product = torch.matmul(features, features.detach().T) / temperature
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        logits_mask = torch.eye(size, dtype=torch.bool, device=features.device).logical_not()
        mask = cls_mask * logits_mask
        logits_mask = logits_mask.float()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        _filter = mask.bool().any(dim=-1)

        # if _filter.any():
        mask = mask[_filter]
        log_prob = log_prob[_filter]

        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        loss = torch.relu(loss)
        # else:
        #     loss = 0

        if meter is not None:
            meter[name] = loss

        return cls_mask, loss

    def loss_mt_contrast_v2_(self, features: torch.Tensor, pys: torch.Tensor,
                             temperature=0.5,
                             meter: Meter = None, name: str = 'Lmtt'):
        """
        选择同类的做正例，类别有 pys 指定（伪标签）
        :param features: [batchsize, feature_dim]
        :param cls_mask:
        :param meter:
        :param name:
        :return:
        """
        size = pys.shape[0]
        pys_ = pys.unsqueeze(-1).repeat([1, size])
        cls_mask = (pys == pys_).float()

        dot_product = tricks.euclidean_dist(features, features, 0)
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        logits_mask = torch.eye(size, dtype=torch.bool, device=features.device).logical_not()
        mask = cls_mask * logits_mask
        logits_mask = (logits_mask * (mask.logical_not())).float()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        _filter = mask.bool().any(dim=-1)

        mask = mask[_filter]
        log_prob = log_prob[_filter]

        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()
        loss = torch.relu(loss)

        if meter is not None:
            meter[name] = loss

        return cls_mask, loss

    def loss_mt_contrast_v3_(self, features: torch.Tensor, pys: torch.Tensor,
                             temperature=1.4,
                             meter: Meter = None, name: str = 'Lmtt'):
        """
        选择同类的做正例，类别有 pys 指定（伪标签）
        :param features: [batchsize, feature_dim]
        :param cls_mask:
        :param meter:
        :param name:
        :return:
        """
        # features = F.normalize(features)

        size = pys.shape[0]
        pys_ = pys.unsqueeze(-1).repeat([1, size])
        cls_mask = (pys == pys_).float()

        dist_matrix = tricks.euclidean_dist_v2(features, features, 0)

        neg_mask = torch.eye(size, dtype=torch.bool, device=features.device).logical_not()
        pos_mask = cls_mask * neg_mask
        neg_mask = (neg_mask * (pos_mask.logical_not())).float()

        neg_dists = dist_matrix * neg_mask
        pos_dists = dist_matrix * pos_mask

        loss = (pos_dists.sum(1) / pos_mask.sum(1) - neg_dists.sum(1) / neg_mask.sum(1) + temperature).relu().mean()

        if meter is not None:
            meter[name] = loss

        return cls_mask, loss

    def loss_one_contrast_(self, features: torch.Tensor, nys: torch.Tensor,
                           temperature=0.5,
                           meter: Meter = None, name: str = 'Lctt'):
        """
        一次选一个有多个类的进来，做对比学习
        :param features: [batchsize, feature_dim]
        :param cls_mask:
        :param meter:
        :param name:
        :return:
        """
        i = 0
        size, = nys.shape
        cls_mask = nys == nys[i]
        while cls_mask.float().sum() == 1 and i < size:
            i += 1
            cls_mask = nys == nys[i]
        if i == size:  # 不存在正例，不能计算loss（loss 会为负值）
            loss = 0
        else:
            anchor = features[i:i + 1, :]
            dot_product = torch.matmul(anchor, features.T) / temperature
            logits = dot_product - dot_product[0][0]

            mask = cls_mask.unsqueeze(0)
            logits_mask = torch.zeros_like(mask)
            logits_mask[i] = 1
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        if meter is not None:
            meter[name] = loss

        return loss

    def loss_shade_contrast(self, features: torch.Tensor,
                            nys: torch.Tensor, ptargets: torch.Tensor, weight: torch.Tensor,
                            temperature=0.5,
                            high_conf_thresh=0.85,
                            meter: Meter = None, name: str = 'Lsct'):
        """
        一个渐变的对比损失，一开始只去试图尽可能的将同类的聚在一起

        之后当类别信息逐渐明显时（置信度变高时），将其一并加入 mask


        :param features: [bsz, 2, feature_dim]
        :param nys: 根据置信度的噪音标签, [bsz]
        :param ptargets: 预测值, [bsz*2, cls_dim]
        :param weight: 筛选权重, [bsz*2]
        :param T:
        :param meter:
        :param name:
        :return:
        """
        size = nys.shape[0]
        nys_ = nys.unsqueeze(-1)
        cls_mask = (nys_ == nys_.T)
        nys = torch.cat([nys, nys])

        cls_mask = cls_mask.repeat(1, 2)

        # 得到样本及其自身增广样本的 mask
        _self_mask = torch.eye(size, dtype=torch.bool, device=features.device).repeat([1, 2])

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        dot_product = torch.matmul(anchor, contrast_features.detach().T) / temperature  # # 两两之间的相似度

        # 得到除了自身之外的 mask
        other_mask = torch.scatter(torch.ones_like(dot_product, dtype=torch.bool),
                                   dim=1,
                                   index=torch.arange(size, device=features.device).view(-1, 1), value=0)
        # 得到自身样本的增广 mask
        _aug_mask = other_mask * _self_mask

        ## 构建正例 mask，高概率置信度 + 类别标签相同的样本为同类，最少也会将增广样本视为正例
        n_targets = onehot(nys, ptargets.shape[1])
        n_targets[weight < 0.5] = ptargets[weight < 0.5]
        _values, _ = ptargets.max(dim=-1)

        _high_conf_mask = (_values > high_conf_thresh).unsqueeze(0).repeat([anchor.shape[0], 1])

        # 所有满足高置信度且类别标签相同的样本的 mask
        same_mask = (cls_mask * _high_conf_mask) | _aug_mask

        # 确保自身不在 mask 中
        pos_mask = (same_mask * other_mask)

        ## 构建异类 mask，同类的 logical_not mask
        neg_mask = (same_mask.logical_not() * other_mask)

        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        loss = - ((pos_mask * log_prob).sum(1) / pos_mask.sum(1)).mean()
        loss = torch.relu(loss)

        if meter is not None:
            meter[name] = loss

        return loss, pos_mask, neg_mask


class PencilLoss(Loss):
    def loss_ce_with_lc_targets_(self, logits: torch.Tensor, targets: torch.Tensor, meter: Meter, name: str = 'Llc'):
        meter[name] = torch.mean(F.softmax(logits, dim=1) * (F.log_softmax(logits, dim=1) - torch.log((targets))))
        return meter[name]

    def loss_ent_(self, logits: torch.Tensor, meter: Meter, name='Lent'):
        meter[name] = - torch.mean(torch.mul(torch.softmax(logits, dim=1), torch.log_softmax(logits, dim=1)))
        return meter[name]


class FixMatchLoss(Loss):
    def loss_ce_with_masked_(self,
                             logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                             meter: Meter, name: str):
        meter[name] = (F.cross_entropy(logits, labels, reduction='none') * mask).mean()
        return meter[name]

    def loss_ce_with_targets_masked_(self,
                                     logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                                     meter: Meter, name: str):
        meter[name] = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1) * mask)
        return meter[name]


class ICTLoss(Loss):
    def loss_mixup_sup_ce_(self, mixed_logits, labels_a, labels_b, lam, meter: Meter, name: str = 'Lsup'):
        loss = lam * F.cross_entropy(mixed_logits, labels_a) + (1 - lam) * F.cross_entropy(mixed_logits, labels_b)
        meter[name] = loss
        return loss

    def loss_mixup_unsup_mse_(self, input_logits, target_logits, decay, meter: Meter, name: str = 'Lunsup'):
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        loss = F.mse_loss(input_softmax, target_softmax)
        meter[name] = loss * decay
        return meter[name]

    def ict_mixup_(self, imgs: torch.Tensor, labels: torch.Tensor, mix=True):
        if mix:
            lam = np.random.beta(1.0, 1.0)
        else:
            lam = 1

        index = np.random.permutation(imgs.shape[0])

        mixed_x = lam * imgs + (1 - lam) * imgs[index, :]
        y_a, y_b = labels, labels[index]
        return mixed_x, y_a, y_b, lam

    def mixup_unsup_(self, imgs: torch.Tensor, logits: torch.Tensor, mix=True):
        '''
        Compute the mixup data. Return mixed inputs, mixed target, and lambda

        :param imgs:
        :param logits: 这里的preds 因为要混合，所以存在一个问题，是preds好还是logits好？
        :param mix:
        :return:
        '''
        if mix:
            lam = np.random.beta(mix, mix)
        else:
            lam = 1.

        index = torch.randperm(imgs.shape[0])
        mixed_x = lam * imgs + (1 - lam) * imgs[index, :]
        mixed_y = lam * logits + (1 - lam) * logits[index]

        return mixed_x, mixed_y, lam


class MentorLoss(Loss):
    def parse_dropout_rate_list_(self):
        """Parse a comma-separated string to a list.

        The format follows [dropout rate, epoch_num]+ and the result is a list of 100
        dropout rate.

        Args:
          str_list: the input string.
        Returns:
          result: the converted list
        """
        str_list = np.array([0.5, 17, 0.05, 78, 1.0, 5])
        values = str_list[np.arange(0, len(str_list), 2)]
        indexes = str_list[np.arange(1, len(str_list), 2)]

        values = [float(t) for t in values]
        indexes = [int(t) for t in indexes]

        assert len(values) == len(indexes) and np.sum(indexes) == 100
        for t in values:
            assert t >= 0.0 and t <= 1.0

        result = []
        for t in range(len(str_list) // 2):
            result.extend([values[t]] * indexes[t])
        return result

    def loss_ce_with_weight_(self):
        pass

    def mentor_mixup_(self, xs: torch.Tensor, targets: torch.Tensor,
                      weight: np.ndarray,
                      beta: float = 8.0):
        """
        MentorMix method.
        :param xs: the input image batch [batch_size, H, W, C]
        :param targets: the label batch  [batch_size, num_of_class]
        :param weight: mentornet weights
        :param beta: the parameter to sample the weight average.
        :return: The mixed images and label batches.
        """
        with torch.no_grad():
            if beta <= 0:
                return xs, targets

            idx = np.arange(xs.shape[0])
            idx = np.random.choice(idx, idx.shape[0], p=weight)
            xs_b = xs[idx]
            targets_b = targets[idx]

            mix = np.random.beta(beta, beta, xs.shape[0])
            mix = np.max([mix, 1 - mix], axis=0)
            mix[np.where(weight > 0.5)] = 1 - mix

            mixed_xs = xs * mix + xs_b * (1 - mix)
            mixed_targets = targets * mix[:, :, 0, 0] + targets_b * (1 - mix[:, :, 0, 0])
            return mixed_xs, mixed_targets


class IEGLossMixin(MixMatchLoss):
    def get_model(self):
        raise NotImplementedError()

    def create_metanet(self):  # -> Tuple[MetaModule, MetaSGD]:
        raise NotImplementedError()

    def loss_regularization_(self, model, l2_decay=1e-5,
                             meter: Meter = None, name: str = 'Lreg'):
        from thexp.contrib import ParamGrouper
        loss = torch.sum(torch.pow(ParamGrouper(model).kernel_params(with_norm=False), 2)) * l2_decay
        if meter is not None:
            meter[name] = loss
        return loss

    def loss_softmax_cross_entropy_with_targets_(self, logits: torch.Tensor, targets: torch.Tensor,
                                                 meter: Meter = None,
                                                 name: str = 'Lcet'):
        loss = -torch.mean(targets * torch.log_softmax(logits, dim=1))
        if meter is not None:
            meter[name] = loss
        return loss

    def weighted_loss_(self, logits: torch.Tensor, targets: torch.Tensor,
                       weighted: torch.Tensor,
                       meter: Meter = None, name: str = 'Lwce'):
        """带权重的损失"""
        loss_ = (targets * torch.log_softmax(logits, dim=1)) * weighted
        loss_ = torch.sum(loss_)
        if meter is not None:
            meter[name] = loss_
        return loss_

    def _ieg_unsupvised_loss(self,
                             image, aug_image, val_image,
                             noisy_label, noisy_true_label, val_label,
                             meter: Meter):
        logits = self.to_logits(image)
        aug_logits = self.to_logits(aug_image)

        guess_targets = self.semi_mixmatch_loss(
            val_image, val_label,
            image, aug_image,
            un_logits=logits, un_aug_logits=aug_logits,
            un_true_labels=noisy_true_label, meter=meter)

        self.loss_kl_ieg_(logits, aug_logits, meter)  # + all_loss

        return logits, aug_logits, guess_targets

    def meta_optimize_(self,
                       noisy_images, noisy_labels, guess_targets,
                       clean_images, clean_labels,
                       meter: Meter):
        device = noisy_images.device
        batch_size = noisy_images.shape[0]
        metanet, metasgd = self.create_metanet()  # type: MetaModule,MetaSGD
        noisy_logits = metanet(noisy_images)
        noisy_targets = onehot(noisy_labels, guess_targets.shape[-1])
        eps_0 = torch.zeros([guess_targets.shape[0], 1],
                            dtype=torch.float,
                            device=device) + 0.9
        noisy_mixed_targets = eps_0 * noisy_targets + (1 - eps_0) * guess_targets
        noisy_loss = -torch.mean(noisy_mixed_targets * torch.log_softmax(noisy_logits, dim=1))

        weight_0 = torch.zeros(guess_targets.shape[0], dtype=torch.float, device=device) + (1 / batch_size)
        lookahead_loss = torch.sum(noisy_loss * weight_0) + self.loss_regularization_(self.model, 'l2_loss')

        val_grads = torch.autograd.grad(lookahead_loss, metanet.params())
        metasgd.meta_step(val_grads)

        val_logits = metanet(clean_images)
        val_targets = onehot(clean_labels, val_logits.shape[-1])
        val_meta_loss = -torch.mean(torch.sum(
            F.log_softmax(val_logits, dim=1) * val_targets,
            dim=1)) + self.loss_regularization_(metanet, meter, 'metal2_loss')

        meta_grad = torch.autograd.grad(val_meta_loss, metanet.params())
        weight_grad, eps_grad = torch.autograd.grad(metanet.params(), [weight_0, eps_0], meta_grad)

        # weight_0 - weight_grad - (1/batchsize)
        weight_1 = torch.clamp_min(weight_0 - weight_grad - (1 / batch_size), 0)
        weight_1 = weight_1 / (torch.sum(weight_1) + 1e-5)

        weight_1 = weight_1.detach()
        eps_1 = (eps_grad < 0).float().detach()

        return weight_1, eps_1, noisy_targets

    def loss_kl_ieg_(self, q_logits, p_logits, consistency_factor, meter: Meter, name='Lkl'):
        q = torch.softmax(q_logits, dim=1)
        per_example_kl_loss = q * (
                torch.log_softmax(q_logits, dim=1) - torch.log_softmax(p_logits, dim=1))
        meter[name] = per_example_kl_loss.mean() * q.shape[-1] * consistency_factor
        return meter[name]

    def noisy_ieg_loss_(self,
                        val_images: torch.Tensor,
                        val_label: torch.Tensor,
                        noisy_images: torch.Tensor, noisy_aug_images: torch.Tensor,
                        noisy_labels: torch.Tensor,
                        noisy_true_labels: torch.Tensor,
                        meter: Meter):
        # meter.loss =
        noisy_logits = self.to_logits(noisy_images)

        # other losses TODO
        # 三个损失，MixMatch的两个+一个KL散度损失
        logits, aug_logits, guess_targets = self._ieg_unsupvised_loss(noisy_images, noisy_aug_images, val_images,
                                                                      noisy_labels, noisy_true_labels, val_label,
                                                                      meter=meter)

        weight_1, eps_1, noisy_targets = self.meta_optimize_(noisy_images, noisy_labels, guess_targets,
                                                             val_images, val_label, meter)

        mixed_targets = eps_1 * noisy_targets + (1 - eps_1) * guess_targets
        net_loss1 = self.loss_softmax_cross_entropy_with_targets_(mixed_targets, logits,
                                                                  meter=meter, name='net_loss')

        init_mixed_labels = 0.9 * noisy_targets + 0.1 * guess_targets
        net_loss2 = self.weighted_loss_(noisy_logits, init_mixed_labels,
                                        weight_1,
                                        meter=meter, name='init_net_loss')

        meter.all_loss = meter.all_loss + (net_loss1 + net_loss2) / 2
        meter.all_loss = meter.all_loss + self.loss_regularization_(
            self.get_model(),
            meter=meter, name='l2_loss')


class SimCLRLoss(Loss):
    def loss_sim_(self, features: torch.Tensor, temperature,
                  device,
                  meter: Meter = None,
                  name: str = 'Lsim'):
        """

        :param features:
        :param temperature:
        :param meter:
        :param name:
        :return:
        """
        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).to(device)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        if meter is not None:
            meter[name] = loss

        return loss


class ConsistenceLoss(Loss):

    def loss_kl_(self, input_logits: torch.Tensor, target_logits: torch.Tensor,
                 meter: Meter = None, name: str = 'Lcon'):
        """Takes softmax on both sides and returns KL divergence
        from ICT implement
        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=-1)
        target_softmax = F.softmax(target_logits, dim=1)
        loss = F.kl_div(input_log_softmax, target_softmax)

        if meter is not None:
            meter[name] = loss

        return loss

    def loss_mse_(self, input_logits: torch.Tensor, target_logits: torch.Tensor,
                  meter: Meter = None, name: str = 'Lcon'):
        """Takes softmax on both sides and returns MSE loss
        from ICT implement

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        loss = F.mse_loss(input_softmax, target_softmax, reduction='sum') / input_logits.shape[1]

        if meter is not None:
            meter[name] = loss

        return loss


class SupConLoss(Loss):
    def loss_supcon_(self, features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor = None,
                     temperature=0.07, base_temperature=0.07, contrast_mode='all',
                     meter: Meter = None, name='Lscon'):
        device = features.device

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            assert False

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ELRLoss(Loss):
    def loss_elr_(self, logits: torch.Tensor, target: torch.Tensor,
                  meter: Meter = None, name='Lelr'):
        y_pred = F.softmax(logits, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        elr_reg = ((1 - (target * y_pred).sum(dim=1)).log()).mean()

        if meter is not None:
            meter[name] = elr_reg

        return elr_reg


class MAELoss(Loss):
    def loss_mae_(self, logits, targets, ):
        pass
