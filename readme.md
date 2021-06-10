## agent 

DQN，Double Dueling DQN，PER，PPO，Rainbow：包含keras和torch两个版本。

Max pressure ：最大压力来选取信号灯

## state

Env提供的单个路口的状态包含单个路口四个方向车道上的车辆数和平均速度。

设计的state如下：

1、单个路口四个方向车道上的车辆数：24维

2、单个路口四个方向车道上的车辆数+平均速度：48维

3、单个路口四个方向车道上的车辆数+平均速度+当前时间点的信号灯：49维

4、单个路口四个方向车道上的车辆数+北、东、南、西与之连接的路口的车辆数：24*5维

## reward

整个路口的压力值 = 所有出道路的车辆-仍在进路口的车辆

## action

8维，参考./images/phases.png

## more information

https://github.com/CityBrainChallenge/KDDCup2021-CityBrainChallenge



