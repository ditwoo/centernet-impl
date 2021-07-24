## Run training

```bash
PYTHONPATH=src python3 -m detector.train \
    --train=$TRAIN_DATASET \
    --train-img-dir=$TRAIN_IMAGES \
    \
    --test=$TEST_DATASET \
    --test-img-dir=$TEST_IMAGES \
    \
    --checkpoint=$CHECKPOINT \
    --lr=$LEARNING_RATE \
    \
    --batch-size=$BATCH_SIZE \
    --num-workers=$NUM_WORKERS \
    --validation-period=$VALIDATION_PERIOD \
    --num-epochs=$NUM_EPOCHS \
    --device=$DEVICE \
    --output=$LOGDIR \
    --progress
```