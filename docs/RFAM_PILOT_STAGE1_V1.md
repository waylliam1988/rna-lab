# Rfam家族试点：第1项来源审核

2026-10-28。用户授权最多3族、每项完成commit/push。固定家族RF00001（5S）、RF00005（tRNA）、RF00010（RNaseP_bact_a）；选择在下载与隔离检查前记录，不因后续冲突换族或追模型成绩。

| 家族 | SEED序列数 | 比对列数 | 来源审核结论 |
| --- | ---: | ---: | --- |
| RF00001 | 712 | 230 | 文献/数据库来源有支持，边级与隔离待审 |
| RF00005 | 954 | 118 | 综述真实，但当前SEED谱系/边级来源仍不完整 |
| RF00010 | 458 | 996 | 文献/数据库来源有支持，边级与隔离待审 |

Rfam元数据为15.1（2026-01-16）。三个SS字段均标Published；不把Published当无模型边的充分证明。RF00001来源为5S数据库及核糖体结构论文；RF00010原数据库论文明确说明比较资料；RF00005引用1993综述且SE为Eddy署名，当前对齐来源证据不足，保留未核验。两个有来源支持的家族也尚未获训练准入。

已保存3份官方SEED和3份官方家族元数据（网页端点实际返回JSON，旧文件后缀.html据content-type作为JSON解析），URL、获取时间、字节数与SHA在本地data/raw/rfam_family_pilot_v1/receipts.json；没有下载FULL库。完整SEED用于审核来源及后续家族关系，不是选出全部序列作训练；代表上限仍为每个合格家族5条。

原文核对依据：

- [5S数据库论文](https://academic.oup.com/nar/article/30/1/176/1332647)：原文/图示描述对齐与二级结构；[其结构来源](https://pubmed.ncbi.nlm.nih.gov/11283358/)。
- [tRNA引用](https://pubmed.ncbi.nlm.nih.gov/8256282/)：PubMed明确标Review；不把它当成当前每条SEED边的实验来源。
- [RNase P原数据库论文](https://academic.oup.com/nar/article/27/1/314/1239359)：Introduction/Description支持比较资料、序列比对与二级结构来源。

尝试取得两份Europe PMC fullTextXML均返回HTTP500，失败收据保留；没有重试或声称已读XML。随后以作者原刊可索引正文、PubMed元数据和图示交叉核对；不以综述/数据库论文证明当前所有边正确。

验证：严格Stockholm解析的5项契约通过，检查跨块拼接、SS_cons长度、声明序列数、结束标记、未知碱基不消失。没有挑代表、生成训练标签或读取模型/旧测试分数。下一项只核对家族/clan及精确/历史关系；关系失败不删除前述来源证据。

机器摘要：data/manifests/rfam_family_pilot_sources_v1.json；源选择：rfam_family_pilot_selection_v1.json。

完整回归：135项测试通过；当前原始数据/模型/旧划分未改，未读预测或测试成绩。
