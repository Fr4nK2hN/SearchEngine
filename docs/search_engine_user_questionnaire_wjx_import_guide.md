# 问卷星导入说明

可直接导入的文本文件：

- [search_engine_user_questionnaire_wjx.txt](/Users/frank/Develop/SearchEngine/docs/search_engine_user_questionnaire_wjx.txt)

建议导入方式：

1. 登录问卷星
2. 创建问卷
3. 选择“导入文本”
4. 将 [search_engine_user_questionnaire_wjx.txt](/Users/frank/Develop/SearchEngine/docs/search_engine_user_questionnaire_wjx.txt) 的全部内容复制粘贴进去
5. 完成导入后，检查以下内容是否识别正常：
   - `段落说明`
   - `单选题`
   - `填空题`
6. 导入后建议手动检查问卷开头的体验地址是否显示正常：
   - `https://search-demo.tail53faab.ts.net/`
7. 建议将“您是否已经访问并体验过本系统？”设为必答题，方便后续筛除无效样本
8. 建议在问卷首页插入二维码图片，避免受访者无法点击未备案域名
   - PNG 文件： [search_engine_funnel_qr.png](/Users/frank/Develop/SearchEngine/docs/assets/search_engine_funnel_qr.png)
   - SVG 文件： [search_engine_funnel_qr.svg](/Users/frank/Develop/SearchEngine/docs/assets/search_engine_funnel_qr.svg)
9. 推荐首页文案：
   - `请先扫码体验搜索系统，再返回本问卷继续填写`
   - `体验地址：search-demo.tail53faab.ts.net`
10. 推荐逻辑设置：
   - 如果“您是否已经访问并体验过本系统？”选择“否，尚未体验”
   - 则跳转到结束页，提示先体验系统后再填写

说明：

- 这份文本采用的是更稳妥的导入结构，核心量表题全部使用 `单选题`
- 这样虽然不是“矩阵量表”显示形式，但更容易被问卷星稳定识别
- 在统计分析上，它仍然是标准的 Likert 量表数据
- 问卷开头已经加入了系统体验网址，受访者可以先打开网站，再返回问卷作答
- 现在也已经生成二维码图片，可以直接上传到问卷首页

参考资料：

- 问卷星帮助中心“从文本创建问卷”：<https://www.wjx.cn/help/help.aspx?h=1&helpid=138>
- 问卷星帮助中心“从文本创建考试”（包含单选题、填空题等文本导入说明）：<https://www.wjx.cn/Help/Help.aspx?helpid=252>
