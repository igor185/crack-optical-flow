from mmflow.apis import inference_model, init_model

config_file = 'configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmflow/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.pth
checkpoint_file = 'checkpoints/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
inference_model(model, 'demo/frame_0001.png', 'demo/frame_0002.png')

print("Done")