from torch import optim


def get_optim_and_scheduler(network, optimizer_config):
    params = network.parameters()

    if optimizer_config["optim_type"] == 'sgd':
        optimizer = optim.SGD(params,
                              weight_decay=optimizer_config["weight_decay"],
                              momentum=optimizer_config["momentum"],
                              nesterov=optimizer_config["nesterov"],
                              lr=optimizer_config["lr"])
    elif optimizer_config["optim_type"] == 'adam':
        optimizer = optim.Adam(params,
                               weight_decay=optimizer_config["weight_decay"],
                               lr=optimizer_config["lr"])
    else:
        raise ValueError("Optimizer not implemented")

    if optimizer_config["sched_type"] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=optimizer_config["lr_decay_step"],
                                              gamma=optimizer_config["lr_decay_rate"])
    elif optimizer_config["sched_type"] == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=optimizer_config["lr_decay_step"],
                                                   gamma=optimizer_config["lr_decay_rate"])
    elif optimizer_config["sched_type"] == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=optimizer_config["lr_decay_rate"])
    else:
        raise ValueError("Scheduler not implemented")

    return optimizer, scheduler
