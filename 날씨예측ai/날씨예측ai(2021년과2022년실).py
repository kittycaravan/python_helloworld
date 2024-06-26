import numpy as np
from sklearn.linear_model import LinearRegression

# 작년의 날짜를 기준으로 1부터 365까지의 순서로 날짜 배열 생성
# dates_last_year = np.arange(1, 366).reshape(-1, 1)
dates_last_year = np.arange(1, 365*2 + 1).reshape(-1, 1)

# 작년 한 해의 평균 기온 데이터(임의의 값)
# avg_temperatures_last_year = np.random.randint(low=-10, high=30, size=365).reshape(-1, 1)

avg_temperatures_last_year = np.array([[
-4.2	,
-5	,
-5.6	,
-3.5	,
-5.5	,
-7.4	,
-14.5	,
-14.9	,
-12.2	,
-7.7	,
-6.7	,
-3.9	,
2	,
1.7	,
4.5	,
-4.9	,
-5.5	,
-3.8	,
-6.3	,
-0.1	,
4.5	,
5.5	,
7.6	,
7.3	,
7.9	,
5.3	,
1.7	,
-2.6	,
-8.8	,
-1.6	,
3.2	,
5	,
-5.6	,
-3.2	,
-3.6	,
1.4	,
6.8	,
3.6	,
-3.1	,
-0.9	,
3.4	,
5.5	,
6.5	,
7.6	,
9.5	,
1.2	,
-5.1	,
-8.3	,
-5.8	,
1.6	,
8.8	,
10.8	,
7.8	,
0	,
2.9	,
4.2	,
8.2	,
9.5	,
7.8	,
4.7	,
2.3	,
4.4	,
7.2	,
9	,
6.6	,
6.2	,
7	,
6.6	,
8.8	,
10.2	,
10.5	,
9.2	,
9.1	,
10.6	,
8.4	,
8.8	,
11.7	,
15.3	,
9.9	,
5.8	,
6.2	,
9.1	,
10.8	,
11.9	,
12.6	,
12.1	,
9	,
10.2	,
10.8	,
14.5	,
17.7	,
17.8	,
14.9	,
11.9	,
11.9	,
13.9	,
14.1	,
13	,
13.6	,
12.6	,
15	,
13.8	,
10.4	,
8.4	,
11.6	,
11.3	,
9.6	,
9.9	,
12.2	,
14.8	,
19.1	,
20.9	,
19	,
18.3	,
18.2	,
16.2	,
15.4	,
14.5	,
14.6	,
11.3	,
10.2	,
12.4	,
14.1	,
13.8	,
13.9	,
15.7	,
14.4	,
14.3	,
15.7	,
13.2	,
18.5	,
22	,
22.9	,
24.1	,
22.1	,
18.4	,
15.3	,
17.9	,
19.9	,
18.7	,
16	,
18.6	,
19.5	,
19.4	,
15.3	,
18	,
15.8	,
15.3	,
15.4	,
18.4	,
19.9	,
20.2	,
23.2	,
19.1	,
18.5	,
20.8	,
23	,
21.1	,
23.6	,
25.8	,
24.9	,
21.5	,
24.2	,
24.8	,
24.8	,
22.5	,
24.3	,
21.9	,
20	,
22.3	,
22.9	,
23.6	,
22.8	,
22.4	,
23	,
24.1	,
22.8	,
24.5	,
23.7	,
23.9	,
24.3	,
26.3	,
27.1	,
22.6	,
21.4	,
23.3	,
25.3	,
26.4	,
26	,
25.5	,
25.9	,
25.7	,
28.1	,
28.8	,
29.7	,
29.1	,
28.8	,
29.1	,
29.6	,
26.4	,
27.9	,
30.5	,
31.2	,
31.2	,
31.7	,
31.5	,
31.2	,
31.1	,
30.4	,
29.6	,
30.5	,
29.8	,
27.1	,
26.5	,
28	,
28.9	,
29.4	,
28.1	,
28	,
26.8	,
28.3	,
27.7	,
27.4	,
26.8	,
26.6	,
26.1	,
26.9	,
27.3	,
25.9	,
24.3	,
26	,
26.6	,
23.8	,
24.4	,
22.4	,
23.4	,
25	,
25.5	,
22.4	,
23.6	,
23.7	,
24.2	,
20.5	,
21.4	,
23.5	,
24.3	,
23.2	,
23.1	,
22.3	,
19.6	,
22.2	,
23.5	,
23.6	,
24	,
24.2	,
24.8	,
24.6	,
23.6	,
22.9	,
23.5	,
23.9	,
23.7	,
23.7	,
22	,
20.2	,
20.9	,
21.8	,
21.2	,
22.6	,
21.6	,
21.5	,
20.2	,
20.3	,
21.1	,
20.7	,
22.3	,
22.9	,
23.6	,
18.8	,
19.4	,
18.8	,
20.9	,
19.9	,
15.2	,
17.9	,
19.6	,
19.7	,
19	,
10.7	,
5.6	,
9.6	,
10.9	,
9.5	,
9.8	,
10.8	,
11.4	,
11.4	,
11.7	,
13	,
13.4	,
12.7	,
14.6	,
13.8	,
14.1	,
12.9	,
11.5	,
12.1	,
12.1	,
13	,
14.1	,
15.3	,
8.6	,
4.9	,
4.2	,
5.3	,
4.5	,
7.3	,
11.3	,
10.1	,
8.5	,
7.5	,
11.9	,
12.9	,
10.4	,
10.4	,
2.7	,
-0.6	,
4.1	,
5.7	,
2.9	,
3.4	,
5.1	,
7.6	,
5.7	,
-1.3	,
1.3	,
1.9	,
1.4	,
2.9	,
5.4	,
6.8	,
6.7	,
6.5	,
7	,
6	,
1.2	,
-2.2	,
3.7	,
7.2	,
6.4	,
-5.6	,
-5.7	,
-0.8	,
5.2	,
5.2	,
2.2	,
2.7	,
-0.3	,
-11.7	,
-12.1	,
-7.6	,
-4.1	,
0.4	,
-3.9	,
-6.7    ,
-4.3	,
-1.3	,
-1.9	,
-2.5	,
-2.8	,
-2.2	,
-1.6	,
0.3	,
1.3	,
0.5	,
-7.4	,
-7.7	,
-6.4	,
-4.8	,
-1	,
-2	,
-5.8	,
-5.6	,
-5.3	,
-5	,
-2.7	,
0.3	,
2	,
4.6	,
2.8	,
1.3	,
-1.1	,
-2.7	,
-3.6	,
-2.8	,
-0.7	,
-1.3	,
-3.4	,
-3.4	,
-5.2	,
-6.6	,
-3.9	,
-2.5	,
-0.1	,
1	,
1.4	,
2.9	,
4.8	,
4.8	,
2.5	,
-4.7	,
-7.3	,
-6.2	,
-1.2	,
-1.7	,
-4.8	,
-3	,
-4.5	,
-5.6	,
-2.4	,
3	,
5.4	,
3.7	,
6.8	,
5.8	,
3.5	,
5.8	,
6.9	,
2.3	,
1.4	,
2.8	,
6	,
7.1	,
9.3	,
11.2	,
12.8	,
11.7	,
8.2	,
7.2	,
10.3	,
10.7	,
7.5	,
3.4	,
5.4	,
5	,
7.1	,
6.2	,
8.2	,
12	,
11.8	,
8.1	,
8.2	,
9.7	,
10.8	,
10.9	,
9.2	,
8.2	,
9.6	,
10.3	,
10.3	,
11.7	,
10.9	,
12.9	,
17.8	,
18.2	,
19.6	,
19.3	,
13.1	,
13.2	,
13.3	,
13.2	,
14	,
15.2	,
15.2	,
15.7	,
16.3	,
15	,
17.7	,
18.5	,
20	,
21.9	,
17.7	,
18.2	,
13.4	,
13	,
13.4	,
14.2	,
14.8	,
17.8	,
18.9	,
19.6	,
18.4	,
15.3	,
17.9	,
19.9	,
19.7	,
20.9	,
19.4	,
15.7	,
16.2	,
17.3	,
20	,
20	,
19.9	,
20.6	,
20.1	,
20.3	,
23	,
22.9	,
21	,
18.7	,
20	,
22.3	,
23.1	,
20.8	,
20.7	,
22.1	,
21.5	,
24.5	,
25.6	,
22.7	,
19.7	,
20.6	,
21.2	,
20.9	,
20.9	,
24.6	,
24.6	,
23.6	,
21.5	,
17.5	,
20.3	,
22.5	,
23.3	,
23.4	,
25.2	,
26.5	,
25.9	,
24.3	,
22.6	,
26	,
26.5	,
26.8	,
26.9	,
25.1	,
22.3	,
26.6	,
28.8	,
29.3	,
29.3	,
29.2	,
29.5	,
28.2	,
25.8	,
26.9	,
29.2	,
27	,
26.8	,
25	,
25.7	,
26	,
24.8	,
25.5	,
26.8	,
27.4	,
27.1	,
23.2	,
25.1	,
24.8	,
25	,
27.5	,
28.9	,
29.3	,
29.7	,
30.7	,
30.9	,
27.2	,
28.6	,
26.8	,
27.1	,
28.9	,
29.4	,
28.7	,
28.9	,
26.8	,
25.1	,
24.7	,
25	,
27	,
25.8	,
27.6	,
28.1	,
26.7	,
26	,
25.6	,
26	,
28.2	,
28.1	,
26.8	,
25.4	,
24.8	,
22.2	,
23.8	,
21.5	,
22.6	,
21	,
19.1	,
21.7	,
24	,
24.7	,
24.2	,
22.3	,
19.2	,
21	,
22.8	,
23.2	,
24.3	,
23.6	,
23.3	,
23.8	,
24.3	,
24.6	,
25.2	,
25.7	,
25.6	,
27.1	,
24.6	,
19.9	,
19.4	,
19.8	,
17.5	,
18.5	,
19.6	,
20	,
20.9	,
20.4	,
20.4	,
20.7	,
20.9	,
19.4	,
21.4	,
17.9	,
16.6	,
15.8	,
14.3	,
15.8	,
13.1	,
9.9	,
11.1	,
13.5	,
16.3	,
17.1	,
18.2	,
16.9	,
11.2	,
9.4	,
10.5	,
12.9	,
14	,
15.1	,
14	,
10.6	,
11.3	,
12.9	,
13.2	,
13.6	,
14.8	,
14.9	,
14.9	,
13.1	,
11.1	,
9.3	,
4.9	,
6.5	,
9.6	,
11.4	,
12.9	,
11.4	,
13.5	,
15.6	,
16.3	,
10.3	,
9.7	,
8.1	,
8.5	,
10.2	,
11	,
12.9	,
13.7	,
11.8	,
10.7	,
10.8	,
9.4	,
12.3	,
6.7	,
5.4	,
12.3	,
6.9	,
-5.4	,
-5.4	,
-2.5	,
2.1	,
-3.5	,
-3.1	,
0.9	,
4.9	,
3.7	,
4.5	,
2.6	,
1.6	,
2	,
0	,
-8.4	,
-4.5	,
-6.3	,
-7	,
-9.5	,
-7	,
-3.2	,
-0.1	,
-7.8	,
-11.8	,
-8.4	,
-5.4	,
-3.9	,
-2.6	,
-3.3	,
-2.9	,
-1.8	,
-1.2	
]]).reshape(-1, 1)

# 선형 회귀 모델 초기화
model = LinearRegression()

# 선형 회귀 모델에 데이터 학습
model.fit(dates_last_year, avg_temperatures_last_year)

# 올해 1월 1일의 날짜 데이터(366일째)
jan_1_date_this_year = np.array([[365+365+1]])

# 모델을 사용하여 올해 1월 1일의 날씨 예측
predicted_temperature_jan_1 = model.predict(jan_1_date_this_year)

# 소수점 이하 자리를 한 자리로 제한
predicted_temperature_jan_1_rounded = round(predicted_temperature_jan_1[0][0], 1)

print("올해 1월 1일의 예상 평균 기온:", predicted_temperature_jan_1_rounded, "도")
