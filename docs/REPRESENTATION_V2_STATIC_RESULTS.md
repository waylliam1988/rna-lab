# 输入v2：static结果

目标、划分、种子、预算和模型尺寸固定；表中主误差为四个未来间隔等权的组宏平均。仅5个开发验证组，无独立测试。

| 模型 | 输入条件 | 主误差均值 ±种子标准差 |
| --- | --- | --- |
| ghost | drop_static_no_geom | 3.578038 ± 0.0916 |
| ghost | keep_static_no_geom | 3.461327 ± 0.125 |
| mlp | drop_static_no_geom | 3.357662 ± 0.0785 |
| mlp | keep_static_no_geom | 3.476388 ± 0.184 |

## 配对贡献

正收益表示新条件误差降低，负值表示误差增加；drop_static的“新条件”是移除第2/3项，add_geometry的新条件是添加实时几何接触。

| 模型 | 比较 | 平均配对误差下降 | 新条件较好种子 | 较好组 | 达到预声明较一致收益规则 |
| --- | --- | --- | --- | --- | --- |
| ghost | drop_static_without_geometry | -3.43% | 0/3 | 0/5 | False |
| mlp | drop_static_without_geometry | 3.20% | 2/3 | 2/5 | False |

## 全部未来间隔

| 模型 | 输入条件 | 未来步数 | 标准化动态MSE | 对线性Skill |
| --- | --- | --- | --- | --- |
| ghost | drop_static_no_geom | 1 | 1.669862 | -2.6612 |
| ghost | drop_static_no_geom | 4 | 2.107344 | 0.2383 |
| ghost | drop_static_no_geom | 8 | 3.146884 | 0.1897 |
| ghost | drop_static_no_geom | 16 | 7.388062 | 0.2377 |
| ghost | keep_static_no_geom | 1 | 1.732367 | -2.7982 |
| ghost | keep_static_no_geom | 4 | 1.975395 | 0.2860 |
| ghost | keep_static_no_geom | 8 | 2.986767 | 0.2310 |
| ghost | keep_static_no_geom | 16 | 7.150781 | 0.2621 |
| mlp | drop_static_no_geom | 1 | 1.382017 | -2.0301 |
| mlp | drop_static_no_geom | 4 | 1.996585 | 0.2783 |
| mlp | drop_static_no_geom | 8 | 2.987887 | 0.2307 |
| mlp | drop_static_no_geom | 16 | 7.064159 | 0.2711 |
| mlp | keep_static_no_geom | 1 | 1.608314 | -2.5262 |
| mlp | keep_static_no_geom | 4 | 2.046863 | 0.2602 |
| mlp | keep_static_no_geom | 8 | 3.051363 | 0.2143 |
| mlp | keep_static_no_geom | 16 | 7.199012 | 0.2572 |

初始权重逐种子配对一致，训练查询顺序和参数量一致；移除通道采用标准化后整列置0，不移除索引12/13静态摘要。没有恢复J门控。几何接触只进入输入，预测目标/损失不增加。原量纲逐通道、逐组/种子、线性基线及权重哈希见各JSON。未检测到收益不等于普遍无用，也不等于统计等价。

解释：Ghost移除第2/3项后的主误差3.5780，高于保留时3.4613，平均配对误差增加3.43%；三个种子、五个组（组内先平均种子）均支持保留，达到预声明的较一致开发收益规则。MLP移除后平均误差下降3.20%，但仅2/3种子、2/5组更好，不构成一致收益。当前证据支持这两项对本轮Ghost有小幅条件贡献，不支持对所有架构一概删除或宣称普遍跨RNA泛化。仍有另外两项静态摘要保留。

三个种子各4个检查点，共12个，已独立重载权重及标准化参数复算一致。下一步保持该结果冻结，添加实时几何接触完成另外两种输入条件。
