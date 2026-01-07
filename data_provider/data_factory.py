from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_Custom


def data_provider(args, flag):
    """
    Data provider factory function.

    Args:
        args: Arguments containing data configuration
        flag: 'train', 'val', or 'test'

    Returns:
        data_set: Dataset instance
        data_loader: DataLoader instance
    """
    size = [args.seq_len, args.label_len, args.pred_len]

    # For CI mode, target_feature should be set
    target_feature = getattr(args, 'target_feature', None)

    data_set = Dataset_Custom(
        root_path=args.root_path,
        flag=flag,
        size=size,
        data_path=args.data_path,
        features=args.features,
        target_features=args.target_features,
        mode=args.mode,
        target_feature=target_feature,
        scale=args.scale,
        timeenc=args.timeenc,
        freq=args.freq
    )

    batch_size = args.batch_size if flag == 'train' else args.batch_size
    shuffle_flag = (flag == 'train')
    drop_last = (flag == 'train')

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
