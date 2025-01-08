import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 1. 指定事件文件路径（请改成你自己的）
event_path = "qm7tut/lightning_logs/version_0/events.out.tfevents.1735555984.node8.891117.0"

# 2. 初始化
ea = event_accumulator.EventAccumulator(event_path)
ea.Reload()  # 载入数据

# 3. 获取你在训练过程中 log 的标量名字
#    例如，你在 LightningModule 里用了 self.log("val_loss",'train_loss',"train_ccsd_pol_MAE","val_ccsd_pol_MAE")
#    那就可以在这里用 "val_loss" 读取
tag_name = "train_ccsd_pol_MAE"
scalars = ea.Scalars(tag_name)

# 4. 解析出 step 和对应的数值
steps = [s.step for s in scalars]
values = [s.value for s in scalars]

# 5. 手动绘制
plt.plot(steps, values, label=tag_name)
plt.xlabel("Step")
plt.ylabel(tag_name)
plt.legend()
plt.savefig('%s.png'%tag_name,dpi=360)
