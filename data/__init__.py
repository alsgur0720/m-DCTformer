from importlib import import_module


from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            # importlib module - import_module function ("package name")
            print(args.data_train)
            module_train = import_module('data.' + args.data_train.lower())
            # getattr(object, attribute) // getattr(c, 'x') = c.x
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'urban100', 'Manga109', 'DIV2K','real', 'real_2','real_3','Idle']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test, train=False)
        else:
            module_test = import_module('data.' +args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, name=args.data_test ,train=False)

        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=not args.cpu
        )

