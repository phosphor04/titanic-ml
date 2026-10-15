from omegaconf import OmegaConf
config = {
    'general': {
        'seed': 42,
        'num_classes': 2,
    },

    'paths': {
        'path_to_train_data': 'data/raw/train.csv',
        'path_to_test_data': 'data/raw/test.csv',
    },

    'split': {
        'n_splits': 5,
        'shuffle': True,
    },

    'preprocessing': {
        'feature_engineering': True, #False
        'data_gaps': True,
        'cat_features_processing': True,
        'age_binning': True,
        'scaler': 'StandardScaler', # StandardScaler or None
    },

    'model': {
        'name': 'LogisticRegression', # LogisticRegression or LGBMClassifier


        'LogisticRegression' : {
            'l1_ratio' : 0.6,
            'max_iter' : 3000,
            'solver' : 'saga',
            'C' : 5
        },

        'KNN': {
            'n_neighbors': 3,
            'weights' : 'uniform',
            'metric' : 'euclidean',
        },

        'DecisionTreeClassifier' : {
            'max_depth' : 3,
            'criterion' : 'gini',
            'random_state' : '${general.seed}',
        },

        'RandomForestClassifier' : {
            'n_estimators' : 75,
            'max_depth' : 7,
            'max_features' : 'sqrt',
            'bootstrap' : True,
            'random_state' : '${general.seed}',
        },

        'CatBoostClassifier': {
            'iterations': 500,
            'learning_rate': 5e-2,
            'depth': 6,
            'random_seed': '${general.seed}',
            'verbose': False
        },

        'LGBMClassifier': {
            'n_estimators': 500,
            'learning_rate': 5e-2,
            'max_depth': 6,
            'num_leaves': 8,
            'min_child_samples' : 17,
            'reg_lambda' : 0,
            'reg_alpha' : 0,
            'colsample_bytree': 1,
            'subsample': 1,
            'subsample_freq': 0,
            'random_state': '${general.seed}',
            'verbosity': -1,
        },

        'XGBClassifier': {
            'n_estimators': 500,
            'learning_rate': 5e-2,
            'max_depth': 2,
            'min_child_weight': 1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0,
            'reg_lambda': 0.5,
            'random_state': '${general.seed}',
        }

    },
}

config = OmegaConf.create(config)

# region Полноценный конфиг с настройками обучения, оптимизатора, scheduler'а и т.д.
# from omegaconf import OmegaConf

# config = {
#     'general': {
#         'seed': 0xFACED,
#         'num_classes': 2, 
#     },
#     'paths': {
#         'path_to_train_data': '/data/raw/train.csv',
#         'path_to_test_data': '/data/raw/test.csv',
#     },
#     'training': {
#         'num_epochs': 200,
#         'early_stopping_epochs': 100,
#         'lr': 1e-4 / 100,

#         'mixed_precision': True,
#         'gradient_accumulation': False,
#         'gradient_clipping': False,
#         'gradient_accumulation_steps': 8,
#         'clip_value': 2,
        
#         'warmup_scheduler': True,
#         'warmup_epochs': 5,
#         'warmup_multiplier': 100,

#         'debug': False,
#         'number_of_train_debug_samples': 5000,
#         'number_of_val_debug_samples': 1000,
        
#         'device': 'cuda',
#         'save_best': True,
#         'save_last': False,
#         'verbose_plots': False,
#         'resume_from_latest_checkpoint': False,
#     },
#     'dataloader_params': {
#         'batch_size': 64,
#         'num_workers': 8,
#         'pin_memory': False,
#         'persistent_workers': True,
#         'shuffle': True,
#         'drop_last': True,
#     },
#     'split': {
#         'all_data_train': False,
#         'n_splits': 5,
#         'folds_to_train': [2, 3, 4, 0, 1],
#         'folds_to_submit': [2, 3, 4, 0, 1],
#         'already_split': False,
#     },
#     'model': {           
#         'freeze_batchnorms': False,
#         'converter_sample_input_shape': [384, 543, 3],
#         'model_sample_input_shape': [384, 828],
#         'params': {
#             "max_length": 384,
#             "embed_dim": 512, 
#             "num_point": 828,
#             "num_head": 16,
#             "num_class": '${general.num_classes}',
#             "num_block": 1, 
#         },
#     },
#     'optimizer': {
#         'name': '/custom/Ranger',
#         'params': {
#             'lr': '${training.lr}',
#         },
#     },
#     'scheduler': {
#         'name': 'CosineAnnealingLR', 
#         'interval': 'epoch',
#         'params': {
#             'T_max': 195,
#             'eta_min': 1e-8,
#         },
#     },
#     'loss': {
#         'name': '/custom/ArcFaceLoss',
#         'params': {
#             's': 45,
#             'm': 0.4,
#             'crit': 'bce',
#             'class_weights_norm': "batch",
#         }
#     },
#     'metric': {
#         'name': 'accuracy_score',
#         'params': {
#         },
#     },
# }

# config = OmegaConf.create(config)

# endregion