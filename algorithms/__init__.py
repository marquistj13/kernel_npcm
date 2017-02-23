import logging
from npcm import npcm
from npcm_plot import npcm_plot
from kernel_npcm import kernel_npcm
from kernel_test import kernel_test
from kernel_npcm_zero import npcm_kernel_zero

# log setup
logger = logging.getLogger('algorithm')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('logging.log', 'w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)