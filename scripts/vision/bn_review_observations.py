"""Explicit human-readable notes from this diagnostic's own rendered crops.

One token per displayed prediction crop (LOSS cards have one crop).
No default decisions or automatic completion of missing notes.
"""
NOTES={
 'B':('cabinet_like','框内可见灰或青柜状块体的面、顶部或基座；仅指外形，不认证仿真资产类别'),
 'P':('pole_fragment','框内主要为天空和杆体或横件片段，并非完整设备主体'),
 'S':('sky','框内主要是无明显结构的天空区域'),
 'G':('ground_shadow','框内为地面网格或阴影窄条，不是设备主体'),
 'M':('mixed_structure','框内同时含杆、墙、地面或块体局部，不对应完整单个主体'),
 'U':('unknown','框内仅极薄条带或结构交叠，无法可靠确认具体内容'),
 'E':('empty_roi','预测框裁剪后没有正面积，不能赋予框内内容；保留原预测和指标'),
 'C':('clear_body','目标的主体、顶面及基座或面板可辨，没有明显主体遮挡'),
 'O':('occluded','目标有主体或顶面证据，但前景设备或杆体遮挡部分内容'),
 'T':('truncated','主体或面板局部可辨，但被原图边缘截断'),
}
TEXT='''
1 B B B B B B B P B B
2 B B B S P M P S B G M
3 B B
4 B U B B
5 B M P
6 U
7 B U M E E E E E
8 B U P G M
9 B B S U
10 B U P U B
11 B G
12 B B B P P B
13 B B S U B
14 B B
15 B B B
16 B
17 B B B
18 B
19 B B B M
20 B P G G E E
21 B B B
22 B
23 B B P B
24 B G E E E
25 B B P P M B
26 B B B P M
27 U
28 O
29 C
30 C
31 C
32 C
33 C
34 O
35 T
36 T
37 C
38 T
39 T
40 O
41 T
42 T
43 T
44 T
45 T
46 T
47 T
48 C
49 O
50 T
51 T
52 U
53 T
54 U
55 C
56 C
57 U
58 O
59 U
60 T
61 C
62 O
63 O
64 T
65 O
66 C
67 P G U B P
68 P
69 S P P S M S M M G M M E M
70 P M U
71 M S P P S U G E
72 G
73 S P P S M S M
74 P U
75 P S S S U G G E
76 P G M G E
77 P B P
78 P P G G E E
79 T
80 T
81 O
82 T
83 T
84 U
85 C
86 G
87 G E G
88 M G
89 G G
90 G E
91 G
92 M
93 G M G E
94 O
95 O
96 C
97 O
98 O
99 C
100 U
101 O
102 O
103 T
104 O
105 T
106 U
107 T
108 U
109 T
110 O
111 O
'''

def parse():
    result={}
    for line in TEXT.strip().splitlines():
        n,*codes=line.split();key=f'E{int(n):02}'
        if key in result:raise ValueError('Duplicate explicit observation')
        result[key]=[NOTES[c] for c in codes]
    return result
