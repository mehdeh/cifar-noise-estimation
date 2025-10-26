# نمونه‌های استفاده (Usage Examples)

این فایل شامل نمونه‌های کاربردی برای استفاده از پروژه است.

## 🚀 شروع سریع

### 1. نصب وابستگی‌ها

```bash
cd /home/ubuntu/cifar-noise-estimation
pip install -r requirements.txt
```

### 2. آموزش مدل با تنظیمات پیش‌فرض

```bash
python train.py
```

خروجی در `runs/exp_YYYYMMDD_HHMMSS/` ذخیره می‌شود.

### 3. ارزیابی مدل آموزش دیده

```bash
# مسیر checkpoint را با مسیر واقعی جایگزین کنید
python test.py --checkpoint runs/exp_YYYYMMDD_HHMMSS/checkpoints/best_model.pth
```

---

## 📚 مثال‌های پیشرفته

### آموزش با پارامترهای سفارشی

```bash
python train.py \
    --config config/default.yaml \
    --exp-name noise_est_v1 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001 \
    --sigma-min 0.0 \
    --sigma-max 1.0 \
    --seed 42
```

### ادامه آموزش از checkpoint

```bash
python train.py \
    --resume \
    --resume-path runs/noise_est_v1_20241026_143022/checkpoints/latest.pth \
    --epochs 200
```

### آموزش با محدوده نویز متفاوت

```bash
# نویز کم
python train.py --exp-name low_noise --sigma-min 0.0 --sigma-max 0.5 --epochs 100

# نویز زیاد  
python train.py --exp-name high_noise --sigma-min 0.5 --sigma-max 2.0 --epochs 100

# محدوده کامل
python train.py --exp-name full_range --sigma-min 0.0 --sigma-max 2.0 --epochs 150
```

### ارزیابی با تحلیل سطح نویز

```bash
python test.py \
    --checkpoint runs/full_range_20241026_150000/checkpoints/best_model.pth \
    --exp-name detailed_eval \
    --evaluate-by-noise-level \
    --sigma-min 0.0 \
    --sigma-max 2.0
```

---

## 🔧 تنظیم فایل کانفیگ

### ایجاد کانفیگ سفارشی

```bash
cp config/default.yaml config/my_config.yaml
```

سپس `config/my_config.yaml` را ویرایش کنید:

```yaml
training:
  epochs: 200
  learning_rate: 0.0005
  batch_size_train: 256

noise:
  sigma_min: 0.0
  sigma_max: 1.5
```

### استفاده از کانفیگ سفارشی

```bash
python train.py --config config/my_config.yaml --exp-name custom_experiment
```

---

## 📊 بررسی نتایج

### نمایش فایل log

```bash
tail -f runs/exp_YYYYMMDD_HHMMSS/logs/train.log
```

### نمایش نمودارها

نمودارها در `runs/exp_YYYYMMDD_HHMMSS/plots/` ذخیره می‌شوند:
- `loss_curve.png`: نمودار loss در طول آموزش
- `samples_epoch_*.png`: نمونه تصاویر در هر epoch
- `test_scatter_predictions.png`: نمودار پراکنش پیش‌بینی‌ها
- `test_noise_distribution.png`: توزیع نویز

---

## 🧪 سناریوهای آزمایشی

### آزمایش 1: تأثیر learning rate

```bash
python train.py --exp-name lr_0001 --lr 0.001 --epochs 50
python train.py --exp-name lr_0005 --lr 0.005 --epochs 50
python train.py --exp-name lr_00001 --lr 0.0001 --epochs 50
```

### آزمایش 2: تأثیر batch size

```bash
python train.py --exp-name bs_64 --batch-size 64 --epochs 50
python train.py --exp-name bs_128 --batch-size 128 --epochs 50
python train.py --exp-name bs_256 --batch-size 256 --epochs 50
```

### آزمایش 3: مقایسه محدوده‌های نویز

```bash
# محدوده کوچک
python train.py --exp-name sigma_0_05 --sigma-min 0.0 --sigma-max 0.5 --epochs 100

# محدوده متوسط
python train.py --exp-name sigma_0_10 --sigma-min 0.0 --sigma-max 1.0 --epochs 100

# محدوده بزرگ
python train.py --exp-name sigma_0_20 --sigma-min 0.0 --sigma-max 2.0 --epochs 100

# ارزیابی هر سه مدل
python test.py --checkpoint runs/sigma_0_05_*/checkpoints/best_model.pth --exp-name eval_0_05
python test.py --checkpoint runs/sigma_0_10_*/checkpoints/best_model.pth --exp-name eval_0_10
python test.py --checkpoint runs/sigma_0_20_*/checkpoints/best_model.pth --exp-name eval_0_20
```

---

## 💡 نکات مهم

### 1. استفاده از GPU

```bash
# به صورت خودکار GPU تشخیص داده می‌شود
python train.py

# یا به صورت صریح
python train.py --device cuda

# استفاده از CPU
python train.py --device cpu
```

### 2. تنظیم seed برای تکرارپذیری

```bash
python train.py --seed 42
python train.py --seed 123
```

### 3. ذخیره checkpoint‌های دوره‌ای

در `config/default.yaml`:

```yaml
training:
  save_freq: 5  # هر 5 epoch ذخیره می‌شود
```

### 4. حل مشکل Out of Memory

```bash
# کاهش batch size
python train.py --batch-size 64

# یا استفاده از CPU
python train.py --device cpu --batch-size 32
```

---

## 📈 نظارت بر آموزش

### استفاده از tmux برای اجرای پس‌زمینه

```bash
# شروع session
tmux new -s training

# اجرای آموزش
python train.py --epochs 200

# جدا شدن: Ctrl+B سپس D

# بازگشت به session
tmux attach -t training
```

### ذخیره خروجی در فایل

```bash
python train.py --epochs 100 2>&1 | tee training_output.log
```

---

## 🎯 مثال کامل: از آموزش تا ارزیابی

```bash
# 1. آموزش مدل
python train.py \
    --exp-name final_model \
    --epochs 150 \
    --lr 0.001 \
    --batch-size 128 \
    --sigma-min 0.0 \
    --sigma-max 1.0 \
    --seed 42

# 2. پیدا کردن بهترین checkpoint
ls runs/final_model_*/checkpoints/best_model.pth

# 3. ارزیابی روی test set
python test.py \
    --checkpoint runs/final_model_20241026_143022/checkpoints/best_model.pth \
    --exp-name final_evaluation \
    --evaluate-by-noise-level

# 4. بررسی نتایج
cat runs/final_evaluation_*/logs/test_metrics.json
```

---

## 🔍 عیب‌یابی

### خطا: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### خطا: CUDA out of memory

```bash
python train.py --batch-size 64  # یا کمتر
```

### خطا: Dataset download fails

```bash
# دانلود دستی CIFAR-10 و قرار دادن در ./datasets/
```

---

## 📞 دستورات مفید

```bash
# نمایش راهنمای train
python train.py --help

# نمایش راهنمای test
python test.py --help

# نمایش ساختار پروژه
find . -name "*.py" | grep -v __pycache__

# پاک کردن تمام نتایج
rm -rf runs/*

# پاک کردن داده‌های دانلود شده
rm -rf datasets/
```

