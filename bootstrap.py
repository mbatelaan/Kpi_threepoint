import numpy as np

startseed = 1234


def bootstrap(raw_data, config_ax, nboot, nbin=1):
    """Take a matrix with the configurations along the config_ax axis and return a bootstrapped version of the matrix with the bootstraps along axis 0

    config_ax = integer of the axis where the configurations are
    nboot = integer number of bootstrap resamples
    nbin = integer number of bins (default 1)
    """
    nconf = int(np.shape(raw_data)[config_ax] / nbin)
    # move the axis with the configurations to axis 0
    data = np.moveaxis(raw_data, config_ax, 0)
    bsdata = np.empty([nboot, *np.shape(data)[1:]])  # Results array
    if nbin > 1:  # sort the data into bins and average over the bin values
        tmpdata = np.array(
            [
                np.average(ien)
                for ien in np.split(
                    np.array(data[0 : nbin * (int(len(data) / nbin))]),
                    self.nconf,
                )
            ]
        )
    else:
        tmpdata = np.array(data)
    myseed = startseed * nconf / nboot
    np.random.seed(int(myseed))

    get_random_ints = np.random.randint
    for iboot in range(nboot):
        random_ints = get_random_ints(0, nconf, nconf)
        bsdata[iboot] = np.average(tmpdata[random_ints], axis=0)
    return bsdata
