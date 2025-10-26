"""
اسکریپت اصلی برای آموزش و ارزیابی مدل تخمین نویز CIFAR-10.

این اسکریپت شامل دو حالت اصلی است:
- train: آموزش مدل ResNet برای تخمین سطح نویز
- test: ارزیابی مدل آموزش‌دیده روی مجموعه آزمون

استفاده:
    python main.py train --config config/default.yaml
    python main.py train --config config/default.yaml --epochs 200 --lr 0.001
    python main.py test --checkpoint runs/exp_20240101_120000/checkpoints/best_model.pth
    python main.py test --checkpoint path/to/checkpoint.pth --config config/custom.yaml
"""

import os
import sys
import click
import torch
import random
import numpy as np

# مسیر src را به path اضافه کن
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.resnet import ResNet18
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.utils.config import (
    load_config,
    save_config,
    create_exp_dir,
    update_config_with_cli
)
from src.utils.logger import Logger


def set_seed(seed):
    """
    تنظیم seed تصادفی برای قابلیت تکرار نتایج.
    
    Args:
        seed (int): مقدار seed تصادفی
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_experiment(config_path, exp_name, cli_args, mode='train'):
    """
    راه‌اندازی آزمایش شامل بارگیری کانفیگ، ایجاد دایرکتوری و logger.
    
    Args:
        config_path (str): مسیر فایل کانفیگ
        exp_name (str): نام آزمایش
        cli_args (dict): آرگومان‌های خط فرمان
        mode (str): حالت اجرا ('train' یا 'test')
    
    Returns:
        tuple: (config, exp_dir, logger)
    """
    # بارگیری کانفیگ پایه
    print(f"بارگیری کانفیگ از: {config_path}")
    base_config = load_config(config_path)
    
    # به‌روزرسانی کانفیگ با آرگومان‌های CLI
    config = update_config_with_cli(base_config, cli_args)
    
    # تنظیم seed تصادفی
    set_seed(config['seed'])
    print(f"Seed تصادفی تنظیم شد: {config['seed']}")
    
    # ایجاد دایرکتوری آزمایش
    exp_dir = create_exp_dir('runs', exp_name)
    print(f"دایرکتوری آزمایش: {exp_dir}")
    
    # ذخیره کانفیگ
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_save_path)
    print(f"کانفیگ ذخیره شد در: {config_save_path}")
    
    # راه‌اندازی logger
    log_dir = os.path.join(exp_dir, 'logs')
    logger = Logger(log_dir, name=mode)
    
    # لاگ عنوان
    title = "آموزش مدل تخمین نویز CIFAR-10" if mode == 'train' else "ارزیابی مدل تخمین نویز CIFAR-10"
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    logger.info(f"دایرکتوری آزمایش: {exp_dir}")
    logger.info(f"فایل کانفیگ: {config_path}")
    
    return config, exp_dir, logger


def create_model_and_log_info(config, logger):
    """
    ایجاد مدل و لاگ اطلاعات آن.
    
    Args:
        config (dict): کانفیگ آزمایش
        logger (Logger): لاگر
    
    Returns:
        torch.nn.Module: مدل ایجاد شده
    """
    logger.info("\nایجاد مدل...")
    model = ResNet18(num_classes=config['model']['num_classes'])
    
    # شمارش پارامترها
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"مدل: {config['model']['name']}")
    logger.info(f"کل پارامترها: {total_params:,}")
    logger.info(f"پارامترهای قابل آموزش: {trainable_params:,}")
    
    return model


@click.group()
@click.version_option(version='1.0.0')
def main():
    """
    ابزار جامع برای آموزش و ارزیابی مدل تخمین نویز CIFAR-10.
    
    این ابزار شامل دو حالت اصلی است:
    - train: آموزش مدل جدید
    - test: ارزیابی مدل آموزش‌دیده
    """
    pass


@main.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='config/default.yaml',
    help='مسیر فایل کانفیگ YAML'
)
@click.option(
    '--exp-name',
    type=str,
    default=None,
    help='نام آزمایش (به timestamp اضافه می‌شود)'
)
@click.option(
    '--epochs',
    type=int,
    default=None,
    help='تعداد epoch های آموزش'
)
@click.option(
    '--batch-size',
    type=int,
    default=None,
    help='اندازه batch آموزش'
)
@click.option(
    '--batch-size-test',
    type=int,
    default=None,
    help='اندازه batch آزمون'
)
@click.option(
    '--lr',
    type=float,
    default=None,
    help='نرخ یادگیری'
)
@click.option(
    '--device',
    type=str,
    default=None,
    help='دستگاه مورد استفاده (cuda یا cpu)'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Seed تصادفی برای تکرارپذیری'
)
@click.option(
    '--sigma-min',
    type=float,
    default=None,
    help='حداقل انحراف معیار نویز'
)
@click.option(
    '--sigma-max',
    type=float,
    default=None,
    help='حداکثر انحراف معیار نویز'
)
@click.option(
    '--resume',
    is_flag=True,
    default=False,
    help='ادامه آموزش از checkpoint'
)
@click.option(
    '--resume-path',
    type=click.Path(exists=True),
    default=None,
    help='مسیر checkpoint برای ادامه آموزش'
)
def train(config, exp_name, epochs, batch_size, batch_size_test, lr, device,
          seed, sigma_min, sigma_max, resume, resume_path):
    """
    آموزش مدل تخمین نویز CIFAR-10.
    
    این دستور یک مدل ResNet را برای پیش‌بینی سطح نویز (sigma) اضافه شده
    به تصاویر CIFAR-10 با استفاده از فرمول x_noise = x + eps * sigma آموزش می‌دهد
    که در آن eps ~ N(0, I).
    """
    # آماده‌سازی آرگومان‌های CLI
    cli_args = {
        'epochs': epochs,
        'batch_size': batch_size,
        'batch_size_test': batch_size_test,
        'lr': lr,
        'device': device,
        'seed': seed,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'resume': resume,
        'resume_path': resume_path
    }
    
    # راه‌اندازی آزمایش
    config, exp_dir, logger = setup_experiment(config, exp_name, cli_args, 'train')
    
    # لاگ کانفیگ
    logger.info("\nکانفیگ:")
    logger.info(f"  Epochs: {config['training']['epochs']}")
    logger.info(f"  اندازه batch (آموزش): {config['data']['batch_size_train']}")
    logger.info(f"  اندازه batch (آزمون): {config['data']['batch_size_test']}")
    logger.info(f"  نرخ یادگیری: {config['training']['learning_rate']}")
    logger.info(f"  بهینه‌ساز: {config['training']['optimizer']}")
    logger.info(f"  زمان‌بند: {config['training']['scheduler']}")
    logger.info(f"  محدوده نویز: [{config['noise']['sigma_min']}, {config['noise']['sigma_max']}]")
    logger.info(f"  دستگاه: {config['device']}")
    logger.info(f"  Seed: {config['seed']}")
    
    # بارگیری داده‌ها
    logger.info("\nبارگیری مجموعه داده CIFAR-10...")
    train_loader, val_loader, _, _ = get_dataloaders(config)
    logger.info(f"نمونه‌های آموزش: {len(train_loader.dataset)}")
    logger.info(f"نمونه‌های اعتبارسنجی: {len(val_loader.dataset)}")
    logger.info(f"batch های آموزش: {len(train_loader)}")
    logger.info(f"batch های اعتبارسنجی: {len(val_loader)}")
    
    # ایجاد مدل
    model = create_model_and_log_info(config, logger)
    
    # ایجاد trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        exp_dir=exp_dir
    )
    
    # آموزش
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\nآموزش توسط کاربر متوقف شد!")
        logger.info("ذخیره checkpoint...")
        trainer.save_checkpoint(trainer.start_epoch + len(trainer.train_losses) - 1)
        logger.info("Checkpoint ذخیره شد. خروج...")
        sys.exit(0)
    
    logger.info("\n" + "=" * 60)
    logger.info("آموزش با موفقیت تکمیل شد!")
    logger.info("=" * 60)
    logger.info(f"نتایج ذخیره شده در: {exp_dir}")
    logger.info(f"بهترین مدل: {os.path.join(exp_dir, 'checkpoints', 'best_model.pth')}")
    logger.info(f"نمودار loss: {os.path.join(exp_dir, 'plots', 'loss_curve.png')}")


@main.command()
@click.option(
    '--checkpoint',
    type=click.Path(exists=True),
    required=True,
    help='مسیر فایل checkpoint مدل'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='config/default.yaml',
    help='مسیر فایل کانفیگ YAML'
)
@click.option(
    '--exp-name',
    type=str,
    default='test',
    help='نام آزمایش برای ذخیره نتایج'
)
@click.option(
    '--batch-size',
    type=int,
    default=None,
    help='اندازه batch آزمون'
)
@click.option(
    '--device',
    type=str,
    default=None,
    help='دستگاه مورد استفاده (cuda یا cpu)'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Seed تصادفی برای تکرارپذیری'
)
@click.option(
    '--sigma-min',
    type=float,
    default=None,
    help='حداقل انحراف معیار نویز برای آزمون'
)
@click.option(
    '--sigma-max',
    type=float,
    default=None,
    help='حداکثر انحراف معیار نویز برای آزمون'
)
@click.option(
    '--save-visualizations/--no-visualizations',
    default=True,
    help='آیا نمودارهای تجسم ذخیره شوند'
)
@click.option(
    '--evaluate-by-noise-level',
    is_flag=True,
    default=False,
    help='ارزیابی عملکرد در سطوح مختلف نویز'
)
def test(checkpoint, config, exp_name, batch_size, device, seed,
         sigma_min, sigma_max, save_visualizations, evaluate_by_noise_level):
    """
    ارزیابی مدل تخمین نویز CIFAR-10.
    
    این دستور یک مدل آموزش‌دیده را بارگیری کرده و عملکرد آن را روی
    مجموعه آزمون ارزیابی می‌کند. همچنین معیارهای دقیق و نمودارهای تجسم تولید می‌کند.
    """
    # آماده‌سازی آرگومان‌های CLI
    cli_args = {
        'batch_size_test': batch_size,
        'device': device,
        'seed': seed,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'checkpoint_path': checkpoint
    }
    
    # راه‌اندازی آزمایش
    config, exp_dir, logger = setup_experiment(config, exp_name, cli_args, 'test')
    config['testing']['checkpoint_path'] = checkpoint
    
    # لاگ اطلاعات
    logger.info(f"Checkpoint: {checkpoint}")
    
    # لاگ کانفیگ
    logger.info("\nکانفیگ:")
    logger.info(f"  اندازه batch: {config['data']['batch_size_test']}")
    logger.info(f"  محدوده نویز: [{config['noise']['sigma_min']}, {config['noise']['sigma_max']}]")
    logger.info(f"  دستگاه: {config['device']}")
    logger.info(f"  Seed: {config['seed']}")
    
    # بارگیری داده‌ها
    logger.info("\nبارگیری مجموعه آزمون CIFAR-10...")
    _, test_loader, _, _ = get_dataloaders(config)
    logger.info(f"نمونه‌های آزمون: {len(test_loader.dataset)}")
    logger.info(f"batch های آزمون: {len(test_loader)}")
    
    # ایجاد مدل
    model = create_model_and_log_info(config, logger)
    
    # ایجاد evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        logger=logger,
        exp_dir=exp_dir
    )
    
    # بارگیری checkpoint
    evaluator.load_checkpoint(checkpoint)
    
    # ارزیابی
    logger.info("\n" + "=" * 60)
    metrics = evaluator.evaluate(save_visualizations=save_visualizations)
    logger.info("=" * 60)
    
    # ارزیابی بر اساس سطح نویز در صورت درخواست
    if evaluate_by_noise_level:
        logger.info("\n" + "=" * 60)
        evaluator.evaluate_by_noise_level(num_bins=10)
        logger.info("=" * 60)
    
    # خلاصه
    logger.info("\n" + "=" * 60)
    logger.info("ارزیابی با موفقیت تکمیل شد!")
    logger.info("=" * 60)
    logger.info(f"نتایج ذخیره شده در: {exp_dir}")
    logger.info(f"معیارها: {os.path.join(exp_dir, 'logs', 'test_metrics.json')}")
    if save_visualizations:
        logger.info(f"نمودارها: {os.path.join(exp_dir, 'plots')}")
    
    # چاپ معیارهای کلیدی
    print("\n" + "=" * 60)
    print("معیارهای کلیدی:")
    print("=" * 60)
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R²:   {metrics['r2_score']:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
