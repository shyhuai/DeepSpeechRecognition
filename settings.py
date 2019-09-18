import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

ak = 'H8BcNhFKMhmlwynyUB4YegOI'
ai = '10469665'
sk = '3272504460441b5bd0d8c43294418b7a'
