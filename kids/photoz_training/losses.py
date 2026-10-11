def mdn_loss(model_output, target):
    distribution = model_output
    return -distribution.log_prob(target).sum()


# def bayesian_mdn_loss(model_output, target, kl_weight):
#     distribution, kl_loss = model_output
#     likelihood_loss = -distribution.log_prob(target).mean()
#     return likelihood_loss + kl_weight * kl_loss
