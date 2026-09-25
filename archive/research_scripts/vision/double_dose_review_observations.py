"""Explicit AI-assisted observations of this round, viewed full-context and own-box crops."""
OBS = {
  "E01": [
    [
      "pole_like",
      "右图缘细长杆体及天空背景被框入，杆体被画面截断，不是完整设备外形。"
    ]
  ],
  "E02": [
    [
      "facade_like",
      "灰色矩形外框、深色面板和底部基座；仅记视觉立面形状，不据此重命名资产。"
    ],
    [
      "facade_like",
      "同一灰色面板结构再次被框入，范围覆盖面板与底座。"
    ],
    [
      "facade_like",
      "第三个预测仍覆盖灰色面板结构，类别变化不改变实际内容。"
    ]
  ],
  "E03": [
    [
      "mixed_structure",
      "右图缘灰色块体的截断侧部、底座和大片网格地面共同入框。"
    ]
  ],
  "E04": [
    [
      "cabinet_like",
      "青色柜状块体正面深色矩形面板、侧壁和黑色底座可辨；资产身份不由外形推断。"
    ]
  ],
  "E05": [
    [
      "cabinet_like",
      "青色柜状结构的面板、右侧壁及底座完整入框。"
    ],
    [
      "cabinet_like",
      "另一seed在同一青色面板与侧壁结构上重复误检。"
    ]
  ],
  "E06": [
    [
      "mixed_structure",
      "灰色块体立面、前景黑杆、青色块体局部重叠入框。"
    ],
    [
      "mixed_structure",
      "同一灰色立面被杆体及青色前景部分遮挡，混合结构重复误检。"
    ]
  ],
  "E07": [
    [
      "facade_like",
      "灰色矩形框内深色面板和底部边界入框。"
    ],
    [
      "facade_like",
      "同一面板被另一类别预测覆盖；仍为灰色面板状立面。"
    ],
    [
      "facade_like",
      "第三个预测重复覆盖同一灰色面板结构。"
    ]
  ],
  "E08": [
    [
      "mixed_structure",
      "灰色侧背立面为主，右缘杆体与青色块体遮挡共同入框。"
    ]
  ],
  "E09": [
    [
      "recognizable_truncated",
      "左下边界截断青色主体，但大面积顶面和侧面可辨。"
    ]
  ],
  "E10": [
    [
      "recognizable_occluded",
      "远处变压器顶面与端子、箱体上半可辨，下部被近处设备遮挡。"
    ]
  ],
  "E11": [
    [
      "recognizable_occluded",
      "灰色柜状主体左部与顶面可见，右下被前景箱体遮挡。"
    ]
  ],
  "E12": [
    [
      "recognizable",
      "圆柱主体、椭圆顶面及基座清晰，未见明显主体遮挡。"
    ]
  ],
  "E13": [
    [
      "recognizable_truncated",
      "右图缘柜体面板及侧壁可辨，主体被右边界截断。"
    ]
  ],
  "E14": [
    [
      "recognizable_occluded_truncated",
      "右边界一列柜体重叠，目标面板局部和顶边可辨，并有前景遮挡。"
    ]
  ],
  "E15": [
    [
      "recognizable_occluded",
      "远处箱体与顶部端子可辨，前景杆体遮挡右侧部分。"
    ]
  ],
  "E16": [
    [
      "unknown_instance_content",
      "目标框落在近处变压器顶面背景后的窄青色区域，独立电容器主体边界不能由RGB可靠确认。"
    ]
  ],
  "E17": [
    [
      "recognizable_truncated",
      "右下图缘青色柜体大面积顶面及主体侧面清楚，下部截断。"
    ]
  ],
  "E18": [
    [
      "recognizable",
      "独立青色长方箱体的顶面、两侧及基座完整可见。"
    ]
  ],
  "E19": [
    [
      "recognizable_occluded",
      "远处灰色电容器块体主体、顶面与基座可辨，左下被青色柜体遮挡。"
    ]
  ],
  "E20": [
    [
      "recognizable_occluded",
      "深青色目标主体与顶面清楚，左下小部分被近处柜体遮挡。"
    ]
  ],
  "E21": [
    [
      "recognizable_truncated",
      "左图缘青色长方主体顶面、侧壁及底座可辨，左侧截断。"
    ]
  ],
  "E22": [
    [
      "recognizable_truncated",
      "大尺度箱体及三个顶部端子清楚，底部触及并被下边界截断。"
    ]
  ],
  "E23": [
    [
      "recognizable_occluded",
      "灰色柜体面板、侧壁和顶面可辨，左下部分被邻柜遮挡。"
    ]
  ],
  "E24": [
    [
      "recognizable",
      "灰色柜体面板轮廓、侧壁、顶面和基座均可辨。"
    ]
  ],
  "E25": [
    [
      "recognizable_truncated",
      "近处灰色变压器大面积顶面与端子可辨，右侧和下侧截断。"
    ]
  ],
  "E26": [
    [
      "recognizable_truncated",
      "深青色变压器顶面、端子和箱体局部清晰，右下图缘截断。"
    ]
  ],
  "E27": [
    [
      "recognizable",
      "远处灰色柜体顶面、正侧轮廓和基座边界清楚。"
    ]
  ],
  "E28": [
    [
      "recognizable_occluded_truncated",
      "左边缘变压器端子和主体可辨，杆体遮挡中部，左侧截断。"
    ]
  ],
  "E29": [
    [
      "recognizable_truncated",
      "近处箱体顶面与三端子清楚，下部被图缘截断。"
    ]
  ],
  "E30": [
    [
      "recognizable_occluded_truncated",
      "右图缘目标柜体的面板局部和顶面可见，邻柜遮挡且边缘截断。"
    ]
  ],
  "E31": [
    [
      "unknown_instance_content",
      "窄框内包含前景变压器顶面和杆体，后方青色薄片是否构成目标主体边界无法可靠分离。"
    ]
  ],
  "E32": [
    [
      "recognizable",
      "大面积矩形主体、底座和斜向阴影清楚，未见主体明显遮挡。"
    ]
  ],
  "E33": [
    [
      "recognizable",
      "背景变色条件下目标矩形大面积侧面和底座清楚，斜影不妨碍外轮廓辨识。"
    ]
  ],
  "E34": [
    [
      "unknown_instance_content",
      "标签窄框与前景变压器顶面端子重叠，后方目标只有薄片状区域，主体归属不足以确认。"
    ]
  ],
  "E35": [
    [
      "recognizable_truncated",
      "左图缘变压器箱体、三端子与基座可辨，左侧被画面截断。"
    ]
  ],
  "E36": [
    [
      "recognizable_occluded",
      "目标矩形主体、顶面和右侧基座可见，左下被前景青色设备遮挡。"
    ]
  ],
  "E37": [
    [
      "limited_occluded",
      "远排柜体仅露顶部及窄侧条，下半被前排柜体遮挡，面板辨识内容不足。"
    ]
  ],
  "E38": [
    [
      "limited_occluded",
      "背景条件下远排目标仅顶部与窄侧条可见，主体下部被前排遮挡。"
    ]
  ],
  "E39": [
    [
      "recognizable_truncated",
      "近处柜体大面积顶面与面板上部可辨，下部图缘截断。"
    ]
  ],
  "E40": [
    [
      "limited_occluded",
      "光照条件下后排柜体只有顶部和狭窄侧条，前排遮挡主体，内容有限。"
    ]
  ],
  "E41": [
    [
      "limited_occluded",
      "后方目标大块侧面及顶边可见，但下部被两层前景箱体遮挡，类别特征有限。"
    ]
  ],
  "E42": [
    [
      "recognizable",
      "灰色大箱体、顶面三端子及底座完整清楚；主体轮廓未见明显遮挡。"
    ]
  ],
  "E43": [
    [
      "limited_occluded",
      "后方目标侧面上部可见，下半被青色块体及近处变压器遮挡，边界受遮挡限制。"
    ]
  ],
  "E44": [
    [
      "recognizable_truncated",
      "右侧远处变压器端子、箱体与底座可辨，右边界截断。"
    ]
  ],
  "E45": [
    [
      "recognizable",
      "青色柜体完整顶面、侧壁、面板和基座清楚。"
    ]
  ],
  "E46": [
    [
      "limited_truncated",
      "右边界仅剩极窄的青色主体和基座切片，无法辨识完整柜体外形。"
    ]
  ],
  "E47": [
    [
      "recognizable_truncated",
      "右图缘柜体面板和侧壁清晰但被右边界截断。"
    ]
  ],
  "E48": [
    [
      "limited_occluded",
      "后方变压器顶面端子与上部侧条可见，下部被前景变压器遮挡，完整主体不足。"
    ]
  ],
  "E49": [
    [
      "recognizable_occluded",
      "变压器端子、顶面与右侧壁可辨，主体中央下方被圆柱电抗器遮挡。"
    ]
  ],
  "E50": [
    [
      "recognizable_truncated",
      "右图缘青色柜体面板、侧壁和基座可辨，右侧截断。"
    ]
  ],
  "E51": [
    [
      "recognizable_occluded_truncated",
      "右侧柜列中目标面板局部、顶部可见，邻柜遮挡并受右图缘截断。"
    ]
  ],
  "E52": [
    [
      "recognizable",
      "圆柱主体、顶面与基座完整清楚，背景颜色不同但主体轮廓可辨。"
    ]
  ],
  "E53": [
    [
      "limited_occluded",
      "最远柜体仅顶部和左侧面板窄片露出，邻柜连续遮挡，特征内容有限。"
    ]
  ],
  "E54": [
    [
      "recognizable",
      "中景独立青色箱体顶面、侧面及底座清楚。"
    ]
  ],
  "E55": [
    [
      "recognizable",
      "灰色变压器完整箱体、三端子和底座可辨。"
    ]
  ],
  "E56": [
    [
      "recognizable",
      "背景条件下青色变压器主体、顶部端子、底座边界清楚。"
    ]
  ],
  "E57": [
    [
      "recognizable",
      "青色方箱主体、顶部及底座完整可见，无明显前景遮挡。"
    ]
  ],
  "E58": [
    [
      "recognizable_occluded",
      "变压器顶面端子及箱体可辨，前景杆体挡住中央偏右部分。"
    ]
  ],
  "E59": [
    [
      "limited_occluded_truncated",
      "左边界目标只有后排顶面与窄侧片，前景箱体遮挡且图缘截断。"
    ]
  ],
  "E60": [
    [
      "limited_occluded_truncated",
      "背景条件下左图缘后排柜体仅有窄顶面和侧片，主体不足。"
    ]
  ],
  "E61": [
    [
      "limited_occluded_truncated",
      "光照条件下左图缘后排目标局部可见，前景箱体遮挡并截断，面板不能辨识。"
    ]
  ],
  "E62": [
    [
      "recognizable",
      "完整箱体、三端子和基座可辨，背景圆柱不遮挡主体。"
    ]
  ],
  "E63": [
    [
      "recognizable",
      "灰色圆柱侧面与矩形基座清楚，顶面受观察角度限制，不等同像素认证。"
    ]
  ],
  "E64": [
    [
      "recognizable",
      "背景条件下圆柱主体与基座轮廓完整可辨，未见明显前景遮挡。"
    ]
  ],
  "E65": [
    [
      "recognizable",
      "光照条件下圆柱侧面渐变和底座清楚，目标主体无遮挡。"
    ]
  ],
  "E66": [
    [
      "recognizable",
      "背景条件下青色柜体面板、侧壁、顶面和基座完整可辨。"
    ]
  ],
  "E67": [
    [
      "limited_occluded",
      "右侧远排柜体仅面板侧片和顶面，前面多台柜体遮挡，内容有限。"
    ]
  ],
  "E68": [
    [
      "recognizable_truncated",
      "右下图缘青色柜体顶面与大面积侧面可辨，下部被画面截断。"
    ]
  ],
  "E69": [
    [
      "recognizable",
      "独立青色长箱体两个侧面、顶面和基座完整清楚。"
    ]
  ],
  "E70": [
    [
      "recognizable",
      "背景条件下独立长箱体顶面、侧面及基座可辨，无明显前景遮挡。"
    ]
  ],
  "E71": [
    [
      "recognizable_occluded",
      "电抗器圆柱顶部和上半侧壁可辨，下部被前景变压器遮挡。"
    ]
  ],
  "E72": [
    [
      "recognizable_occluded",
      "背景条件下圆柱上部与椭圆顶面可辨，下半被前景箱体遮挡。"
    ]
  ],
  "E73": [
    [
      "recognizable_truncated",
      "灰色变压器大面积主体、端子与基座可辨，右侧图缘截断。"
    ]
  ],
  "E74": [
    [
      "recognizable_occluded",
      "中景灰色柜体顶部和主体大部分可辨，右下被前景灰色箱体遮挡，同色边界对比偏低。"
    ]
  ],
  "E75": [
    [
      "limited_occluded",
      "右侧最远柜体只有顶面与面板窄条，多台邻柜遮挡，内容有限。"
    ]
  ],
  "E76": [
    [
      "recognizable_occluded",
      "灰色变压器顶面端子及箱体清楚，前景杆体遮挡中部。"
    ]
  ],
  "E77": [
    [
      "recognizable_occluded",
      "圆柱顶面与上部侧壁可辨，下部被变压器顶面遮挡。"
    ]
  ],
  "E78": [
    [
      "recognizable_truncated",
      "右侧灰色变压器端子、两个主体侧面与基座可辨，右边界截断。"
    ]
  ],
  "E79": [
    [
      "recognizable_occluded",
      "灰色圆柱顶面与上半侧面可辨，下部被前景同色变压器遮挡。"
    ]
  ],
  "E80": [
    [
      "recognizable_truncated",
      "右图缘圆柱主体、顶面局部与基座可辨，右侧截断。"
    ]
  ],
  "E81": [
    [
      "recognizable_truncated",
      "背景条件下右图缘圆柱侧面、顶面局部和底座清楚，右側截断。"
    ]
  ],
  "E82": [
    [
      "limited_occluded",
      "后方变压器端子、顶面及侧壁上条可见，下部被近处变压器遮挡，主体完整性有限。"
    ]
  ],
  "E83": [
    [
      "recognizable_truncated",
      "左图缘灰色箱体的顶面、侧壁及底座清楚，左侧截断。"
    ]
  ],
  "E84": [
    [
      "recognizable_truncated",
      "右图缘灰色变压器端子、侧面与基座可辨，右侧截断。"
    ]
  ],
  "E85": [
    [
      "recognizable_occluded_truncated",
      "左图缘灰色变压器主体和端子可见，前景杆体遮挡中部且左侧截断。"
    ]
  ],
  "E86": [
    [
      "recognizable_occluded",
      "远处灰色变压器端子和主体大部分可辨，下部被前景灰色箱体遮挡。"
    ]
  ],
  "E87": [
    [
      "recognizable_truncated",
      "右下图缘柜体顶面与主体侧面清楚，下部截断。"
    ]
  ],
  "E88": [
    [
      "limited_occluded",
      "柜列目标顶面和侧面上条可见，下部被前排柜体遮挡，无完整面板内容。"
    ]
  ]
}

