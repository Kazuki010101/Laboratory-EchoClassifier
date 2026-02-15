import argparse
import MLPMixer4sig_dist
import PatchEchoClassifier
import PatchEchoAttnClassifier
import torch
import DeepConvLSTM
import resnet4sig

def main():
    parser = argparse.ArgumentParser(description="Example with conditional execution based on arguments.")
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--reservoir_size', type=int, required=False)
    parser.add_argument('--patch_size', type=int, required=False)
    args = parser.parse_args()

    if args.model == "MLPMixer":
        model = MLPMixer4sig_dist.DistilledMLPMixer(dim=512,num_classes=8, depth=8)
        
    elif args.model == "PRC":
        print(f"Creating model: PatchReservoir")
        model = PatchEchoClassifier.PatchReservoir(in_channels=3, patch_size=args.patch_size, stride=args.patch_size, reservoir_size=args.reservoir_size, num_classes=8) #stride=1から変更
    elif args.model == "PEAC":
        print(f"Creating model: PatchAttnReservoir")
        model = PatchEchoAttnClassifier.PatchReservoir(in_channels=3, patch_size=args.patch_size, stride=args.patch_size, reservoir_size=args.reservoir_size, num_classes=8)
    elif args.model == "PEAC_Q":
        print(f"Creating model: PatchAttnReservoir")
        model = PatchEchoAttnClassifier.PatchReservoir(in_channels=3, patch_size=args.patch_size, stride=args.patch_size, reservoir_size=args.reservoir_size, num_classes=8)
        model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8)
    elif "DeepConvLSTM" in args.model:
        if "100" in args.model:
            config= {
            'n_hidden': 128,
            'n_layers': 1,  
            'n_filters': 64,
            'n_classes': 8,  
            'filter_size': 5,  
            'window_size': 496,  
            'channels': 3,  
            'drop_prob': 0.5,  
            }
        elif "50" in args.model:
            config = {
            'n_hidden': 64,  # 128 → 64
            'n_layers': 1,  
            'n_filters': 32,  # 64 → 32
            'n_classes': 8,  
            'filter_size': 5,  
            'window_size': 496,  
            'channels': 3,  
            'drop_prob': 0.5,  
            }
        elif "25" in args.model:
            config = {
            'n_hidden': 32,  # 128 → 32
            'n_layers': 1,  
            'n_filters': 16,  # 64 → 16
            'n_classes': 8,  
            'filter_size': 5,  
            'window_size': 496,  
            'channels': 3,  
            'drop_prob': 0.5,  
            }

        model = DeepConvLSTM.DeepConvLSTM(**config)
    elif "Resnet" in args.model:
        if "L" in args.model:
            model = resnet4sig.ResNet1D(
                in_channels=3,
                base_filters=64,
                kernel_size=7,
                stride=2,
                groups=1,
                n_block=8,
                n_classes=8,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False
                )
        elif "M" in args.model:
            model = resnet4sig.ResNet1D(
                in_channels=3,
                base_filters=32,
                kernel_size=7,
                stride=2,
                groups=1,
                n_block=8,
                n_classes=8,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False
                )

        elif "S" in args.model:
            model = resnet4sig.ResNet1D(
                in_channels=3,
                base_filters=16,
                kernel_size=7,
                stride=2,
                groups=1,
                n_block=4,
                n_classes=8,
                downsample_gap=2,
                increasefilter_gap=4,
                use_bn=True,
                use_do=True,
                verbose=False
                )
        
    batch_size = 64
    channels = 3
    signal_length = 496
    x = torch.randn(batch_size, channels, signal_length)
    #model.to("cuda:0")
    #x=x.cuda()
    
    _ = model(x)


if __name__ == '__main__':
    main()