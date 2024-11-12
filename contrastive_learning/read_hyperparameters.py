from tbparse import SummaryReader
log_dir = "./logs/VanillaVAE_DS3/version_17"
reader = SummaryReader(log_dir)
hp = reader.hparams
print(hp)