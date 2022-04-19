import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from model.yolo import YoloBody
from utils.utils_train import *
from utils.dataloader import YoloDataset, yolo_dataset_collate
from model.utils import read_config

from utils.EarlyStopping import *


def do_train(start_epoch, end_epoch, model_train, yolo_loss, loss_history, epoch_step, epoch_step_val, train_gen,
             valid_gen, save_period, early_stopping):
    for epoch in range(start_epoch, end_epoch):

        train_gen.dataset.epoch_now = epoch
        valid_gen.dataset.epoch_now = epoch

        loss = 0
        model_train.train()
        with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for idx, batch in enumerate(train_gen):
                if idx >= epoch_step:
                    break

                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

                optimizer.zero_grad()
                outputs = model_train(images)

                # ---------------------------------------------------#
                loss_value_all = 0
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets)
                    loss_value_all += loss_item

                loss_value = loss_value_all

                loss_value.backward()
                optimizer.step()

                loss += loss_value.item()

                pbar.set_postfix(**{'loss': loss / (idx + 1), 'lr': get_lr(optimizer)})
                pbar.update(1)

        # ---------------------------------------------------#
        # validation
        val_loss = 0
        model_train.eval()
        with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{end_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for idx, batch in enumerate(valid_gen):
                if idx >= epoch_step_val:
                    break

                images, targets = batch[0], batch[1]

                with torch.no_grad():
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

                    optimizer.zero_grad()
                    outputs = model_train(images)

                    loss_value_all = 0
                    for l in range(len(outputs)):
                        loss_item = yolo_loss(l, outputs[l], targets)
                        loss_value_all += loss_item
                    loss_value = loss_value_all

                val_loss += loss_value.item()
                pbar.set_postfix(**{'val_loss': val_loss / (idx + 1)})
                pbar.update(1)

        # ---------------------------------------------------#
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(end_epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # ---------------------------------------------------#
        # early_stopping
        if early_stopping != None:

            early_stopping(val_loss / epoch_step_val, model_train)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # ---------------------------------------------------#
        save_path = loss_history.log_dir
        if (epoch + 1) % save_period == 0 or epoch + 1 == end_epoch:
            torch.save(model_train.state_dict(), '%s/ep%03d-loss%.3f-val_loss%.3f.pth' % (
                save_path, epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
        if (val_loss / epoch_step_val) == loss_history.min_valid_loss:
            torch.save(model_train.state_dict(), '%s/best.pth' % save_path)

        # ---------------------------------------------------#
        if (epoch + 1) % last_period == 0 or epoch + 1 == end_epoch:
            checkpoint = {'epoch': epoch,
                          'loss': loss_history.loss,
                          'model': model_train.state_dict(),
                          'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, '%s/last.pt' % save_path)
            del checkpoint

        lr_scheduler.step()


if __name__ == "__main__":

    save_period = 10
    last_period = 1

    # ---------------------------------------------------#
    # 초기화
    config = read_config()
    pramas = dict(config['model'])

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    class_names = [name.strip() for name in pramas['class_names'].split(',')]
    num_classes = len(class_names)

    anchors = np.array([float(x) for x in pramas['anchors'].split(',')]).reshape(-1, 2)
    num_anchors = len(anchors)

    model_path = pramas['weight_path']
    input_shape = np.array([int(x) for x in pramas['input_shape'].split(',')])

    # ---------------------------------------------------#
    # 학습 옵션
    mosaic = False
    Cosine_lr = True
    label_smoothing = 0


    # ---------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes)
    init_weights(model)

    save_dir = 'logs'
    loss_history = LossHistory(save_dir, model, input_shape)

    # cuda
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()

    # learning rate
    lr = 1e-4
    optimizer = optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    # ---------------------------------------------------#
    # 1. 완전 다된거(ex. yolov4_weights.pth) : 고려하지 않음.
    # 2. 백본만 가져다 쓸거(ex. cspdarknet53.pth)
    # 3. 학습 재개(last.pt)
    pretrained = True  # 백본 초기화
    resume = False  # 전체 초기화

    # ---------------------------------------------------#
    resume_path = 'logs/loss_2022_04_08_10_02_36/last.pt'

    if resume:
        optimizer, loss, start_epoch, model_train = resume_weights(resume_path, optimizer, model_train)
        loss_history.load_loss(loss)
    elif pretrained:
        load_weights(model, model_path)

    # ---------------------------------------------------#
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, label_smoothing)

    # ---------------------------------------------------#
    # 학습 데이터
    batch_size = 8
    num_workers = 0

    train_path = '2007_train_mini.txt'
    valid_path = '2007_val_mini.txt'

    with open(train_path) as f:
        train_lines = f.readlines()
    with open(valid_path) as f:
        valid_lines = f.readlines()

    train_dataset = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train=True)
    valid_dataset = YoloDataset(valid_lines, input_shape, num_classes, mosaic=False, train=False)
    train_gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                           drop_last=True, collate_fn=yolo_dataset_collate)
    valid_gen = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                           drop_last=True, collate_fn=yolo_dataset_collate)

    # ---------------------------------------------------#
    epoch_step = len(train_lines) // batch_size
    epoch_step_val = len(valid_lines) // batch_size

    """
    # ---------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01

    # ---------------------------------------------------#
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type in ['adam', 'adamw'] else 5e-2
    lr_limit_min = 3e-4 if optimizer_type in ['adam', 'adamw'] else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)   
    """

    # ---------------------------------------------------#

    # ---------------------------------------------------#
    freeze_epoch = 10
    unfreeze_epoch = 20

    # ---------------------------------------------------#
    # freeze training
    # ---------------------------------------------------#
    start_epoch = 0
    end_epoch = freeze_epoch

    early_stopping = EarlyStopping(patience=10, verbose=True)

    for param in model.backbone.parameters():
        param.requires_grad = False

    do_train(start_epoch, end_epoch, model_train, yolo_loss, loss_history, epoch_step, epoch_step_val, train_gen,
             valid_gen, save_period, early_stopping=None)

    # ---------------------------------------------------#
    # unfreeze training
    start_epoch = end_epoch
    end_epoch = unfreeze_epoch

    for param in model.backbone.parameters():
        param.requires_grad = True

    do_train(start_epoch, end_epoch, model_train, yolo_loss, loss_history, epoch_step, epoch_step_val, train_gen,
             valid_gen, save_period, early_stopping)

    # set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    # ---------------------------------------------------#
