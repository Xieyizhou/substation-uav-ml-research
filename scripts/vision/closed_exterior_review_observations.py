"""Explicit observations from all sixteen four-condition cards, not automatic approvals."""
# Each note covers an explicitly viewed full frame and its identified box in all four displayed conditions.
OBS={
 'layout-A:closed:reactor:1':{'reactor_north':'圆柱顶面、完整侧面和基座可辨；右下角未框片段须实例核验，不能按外形自行归属。'},
 'layout-A:closed:reactor:2':{'west_switchgear_01':'斜俯视柜体及面板可辨；前景灰色结构靠近框下缘。','reactor_north':'中距圆柱主体与基座清楚，未见明显主体遮挡。'},
 'layout-A:closed:reactor:3':{'west_switchgear_01':'前景灰结构遮挡下半部，只余上部/面板局部，暂缓整组。','reactor_north':'远距圆柱主体可辨，基座及左下缘被前景遮挡，不能称完整无遮挡。'},
 'layout-A:closed:capacitor_bank:1':{'capacitor_east':'封闭箱体顶面、侧面及基座完整；只作为封闭外壳，不声称外露组件。'},
 'layout-A:closed:capacitor_bank:2':{'west_switchgear_04':'中后景小柜体有可辨顶侧面。','transformer_sw':'右侧主体和套管可见，杆体在前景形成遮挡。','transformer_se':'后方主体与顶部套管可辨。','capacitor_east':'计划箱体下部被灰色前景遮挡，主要剩上部，整组暂缓。'},
 'layout-A:closed:switchgear:1':{'west_switchgear_02':'右后方柜体正面板可辨，变体中对比变弱仍见边缘。','switchgear_west':'左后方横向柜体和面板可辨。','entry_switchgear':'计划目标正面板、外壳及底座清楚，四条件都有面板边界。'},
 'layout-A:closed:switchgear:2':{'west_switchgear_01':'计划柜体斜前方，侧面比例与正面板边界可辨。','reactor_north':'后方圆柱被前景柜体遮住较大下部，仅上段较清楚，整组暂缓。'},
 'layout-A:closed:transformer:1':{'west_switchgear_03':'右侧柜体顶面及侧向面板可辨，无明显截断。','transformer_mid':'俯视主体和三根套管完整可辨，侧立面较少但不是仅基座。'},
 'layout-B:closed:reactor:1':{'reactor_north':'圆柱主体、顶面及基座清楚；旁侧青色柜状物与后方灰结构未框，需来源核验而非按颜色判类。'},
 'layout-B:closed:reactor:2':{'reactor_north':'较低视角圆柱侧面与底座完整，旁侧未框结构需实例证据核验。'},
 'layout-B:closed:reactor:3':{'reactor_north':'较远圆柱主体完整、与杆体分离，旁侧未框结构不能据外观判作目标设备。'},
 'layout-B:closed:capacitor_bank:1':{'capacitor_east':'近距封闭箱体与底座清楚，没有外露组件主张。'},
 'layout-B:closed:capacitor_bank:2':{'transformer_mid':'右侧变压器主体、底座和三根套管完整可辨。','capacitor_east':'较远封闭箱体斜俯视外壳比例与底座可辨。'},
 'layout-B:closed:switchgear:1':{'west_switchgear_03':'远距正面小柜体和面板边缘可辨；暖/冷/中性下对比弱于原始，但不是只有色点。','transformer_sw':'旁侧主体与三个顶部套管清楚，完整框不接图缘。'},
 'layout-B:closed:switchgear:2':{'reactor_north':'左后圆柱主体与底座可辨，旁侧杆体邻近边界。','entry_switchgear':'斜前近距柜体外壳、面板及基座清楚；后方未框青色块需来源核验。'},
 'layout-B:closed:transformer:1':{'transformer_mid':'前景主体、三个套管及基座完整。','entry_switchgear':'左侧柜体与面板可辨。','capacitor_east':'后方箱体下部被计划变压器大幅遮挡，只余上部箱体，整组暂缓。'},
}
HOLDS={'layout-A:closed:reactor:3','layout-A:closed:capacitor_bank:2','layout-A:closed:switchgear:2','layout-B:closed:transformer:1'}
VARIANT_NOTES={
 'original':'已查看原始配色全图；类别判断绑定实例映射，不靠青色/灰色命名。',
 'warm':'已查看暖色全图；主体与允许变化面板均呈暖色，低对比边界在记录中单列。',
 'cool':'已查看冷色全图；主体轮廓可见性按本条件记录，不继承原始条件认证。',
 'neutral':'已查看中性灰全图；面板对比变弱但未用颜色恢复类别身份。',
}
