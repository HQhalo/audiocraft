# from audiocraft.utils import export
# from audiocraft import train
# xp = train.main.get_xp_from_sig('c9ca89f6')
# export.export_encodec(
#     xp.folder / 'checkpoint.th',
#     '/home/ubuntu/checkpoints/encodec/compression_state_dict.bin')
    

    # Exporting .bin files from a training run:

from audiocraft.utils import export
from audiocraft import train

sig = "33fd1f20"


xp = train.main.get_xp_from_sig(sig)
export.export_lm(xp.folder / 'checkpoint.th', '/home/ubuntu/checkpoints/finetune_3/state_dict.bin')
xp_encodec = train.main.get_xp_from_sig('c9ca89f6')
export.export_encodec(xp_encodec.folder / 'checkpoint.th', '/home/ubuntu/checkpoints/finetune_3/compression_state_dict.bin')


# xp = train.main.get_xp_from_sig(sig)
# export.export_lm(xp.folder / 'checkpoint.th', '/home/ubuntu/checkpoints/finetune/state_dict.bin')
# export.export_pretrained_compression_model('facebook/encodec_32khz', '/home/ubuntu/checkpoints/finetune/compression_state_dict.bin')
