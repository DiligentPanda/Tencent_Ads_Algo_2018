# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao
# --------------------------------------------------------

import os
import logging
import shutil

def create_logger(output_path,comment,timestamp):

    # archive_name = "{}_{}.tgz".format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    # archive_path = os.path.join(os.path.join(final_output_path, archive_name))
    # pack_experiment(".",archive_path)

    log_file = '{}_{}.log'.format(timestamp,comment)

    head = '%(levelname)s %(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    return logger
