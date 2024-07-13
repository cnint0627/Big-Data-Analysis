# 绘制三种模型在不同指标上的折现对比图
import matplotlib.pyplot as plt

base = ['基于用户（迷你哈希）', '基于用户', '基于内容','基于内容（迷你哈希）']

y1 = [270.36,251.20,175.73,164.35]
y2 = [2.92,4.83,40.25,10.56]

plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.figure(figsize=(8, 4))
ax1 = plt.subplot(1, 1, 1)  # 子绘图1
plt.xlabel('模型')  # x轴标题
plt.ylabel('指标(%)')  # y轴标题
bar_width = 0.15

# plt.ylim(0.5, 1)  # 设置 y 轴范围
plt.plot(base, y1, marker='o', markersize=5, linestyle='-')  # 绘制折线图，添加数据点，设置点的大小
# plt.plot(base, y4, marker='s', markersize=3, linestyle='-')
plt.legend(['SSE'])  # 设置折线名称

# 创建第二个Y轴
ax2 = ax1.twinx()

# 在第二个Y轴上绘制第二组数据
ax2.plot(base, y2, marker='^', markersize=5, linestyle=':', color='red')
ax2.set_ylabel('训练用时(s)')
ax2.tick_params('y')
plt.legend(['训练用时'], loc='lower right')  # 设置折线名称

plt.suptitle("各模型性能对比")
plt.savefig('img.png')
plt.show()  # 显示折线图
