import torch
from torch.utils.data import DataLoader, SequentialSampler
from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader


def dataloader_youcook_train(args, tokenizer, opt_tokenizer=None):
    max_txt_len = getattr(args, 'max_txt_len', 32)
    youcook_dataset = Youcook_Caption_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        opt_tokenizer=opt_tokenizer,
        max_txt_len=max_txt_len,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(youcook_dataset), train_sampler


def dataloader_youcook_test(args, tokenizer, logger, opt_tokenizer=None):
    max_txt_len = getattr(args, 'max_txt_len', 32)
    youcook_testset = Youcook_Caption_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        opt_tokenizer=opt_tokenizer,
        max_txt_len=max_txt_len,
    )

    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))
    return dataloader_youcook, len(youcook_testset)


def dataloader_msrvtt_train(args, tokenizer, opt_tokenizer=None):
    max_txt_len = getattr(args, 'max_txt_len', 32)
    scst = getattr(args, 'scst', False)
    use_small_dataset = getattr(args, 'use_small_dataset', False)
    msrvtt_dataset = MSRVTT_Caption_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        opt_tokenizer=opt_tokenizer,
        max_txt_len=max_txt_len,
        scst=scst,
        use_small_dataset=use_small_dataset,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, logger=None, split_type="test", opt_tokenizer=None):
    max_txt_len = getattr(args, 'max_txt_len', 32)
    use_small_dataset = getattr(args, 'use_small_dataset', False)
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        opt_tokenizer=opt_tokenizer,
        max_txt_len=max_txt_len,
        use_small_dataset=use_small_dataset,
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


DATALOADER_DICT = {
    "youcook": {
        "train": dataloader_youcook_train,
        "val": dataloader_youcook_test
    },
    "msrvtt": {
        "train": dataloader_msrvtt_train,
        "val": dataloader_msrvtt_test
    }
}
