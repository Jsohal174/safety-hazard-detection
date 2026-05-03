# llava-phi3:latest — cot (nothink)

Date: 2026-03-12 18:12
Test set: 99 images

## Results Summary

**Overall accuracy: 48.1%** (79/99 valid)
**Average time: 3.4s per image**

| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN |
|-------|----------|-----------|--------|----|----|----|----|
| spill | 85.7% | 0.471 | 0.857 | 0.608 | 24 | 27 | 4 |
| improper_stacking | 3.8% | 0.333 | 0.038 | 0.069 | 1 | 2 | 25 |
| safe | 52.0% | 0.520 | 0.520 | 0.520 | 13 | 12 | 12 |

## Confusion Matrix

| Ground Truth \ Predicted | spill | improper_stacking | safe | None |
|---|---|---|---|---|
| spill | 24 | 1 | 3 | 0 |
| improper_stacking | 16 | 1 | 9 | 0 |
| safe | 11 | 1 | 13 | 0 |

---

## Detailed Per-Image Results

### spill_frame_0006.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 50%
- **Time:** 4.9s | TTFT: 2.05s | 108 tokens @ 38.8 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 50%
```

---

### stacking_frame_0064.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 3.1s | TTFT: 2.13s | 38 tokens @ 39.8 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / unsafe
confidence: 80%
```

---

### spill_frame_0100.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.0s | TTFT: 2.46s | 38 tokens @ 40.1 tok/s

**Model response:**
```

QUESTION 1: YES

QUESTION 2: YES

category: spill / improper_stacking / unsafe
confidence: 50%
```

---

### stacking_frame_0054.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.5s | TTFT: 2.17s | 18 tokens @ 40.4 tok/s

**Model response:**
```

QUESTION 1: No.

QUESTION 2: Yes.
```

---

### safe_frame_0043.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.8s | TTFT: 2.2s | 29 tokens @ 40.2 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO
```

---

### stacking_frame_0000_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.0s | TTFT: 2.07s | 38 tokens @ 39.6 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / safe
confidence: 50%
```

---

### spill_frame_0115.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.1s | TTFT: 2.07s | 38 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / unsafe
confidence: 70%
```

---

### safe_frame_0017.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 80%
- **Time:** 2.8s | TTFT: 2.3s | 29 tokens @ 39.8 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: unsafe
confidence: 80%
```

---

### safe_frame_0058.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.4s | TTFT: 2.06s | 16 tokens @ 42.0 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes
```

---

### spill_frame_0022_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 60%
- **Time:** 4.9s | TTFT: 2.08s | 111 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 50-60%
```

---

### spill_frame_0024_v0.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 2.8s | TTFT: 2.06s | 29 tokens @ 39.5 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: unsafe
confidence: 90%
```

---

### spill_frame_0064.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 0%
- **Time:** 3.1s | TTFT: 2.19s | 35 tokens @ 39.6 tok/s

**Model response:**
```

Question 1: Yes

Question 2: Yes

Category: spill / improper_stacking / unsafe
Confidence: 0%
```

---

### stacking_frame_0010_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 65%
- **Time:** 2.6s | TTFT: 2.28s | 22 tokens @ 41.1 tok/s

**Model response:**
```

Category: spill / improper_stacking / safe
Confidence: 65%
```

---

### stacking_frame_0027.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 3.2s | TTFT: 2.37s | 37 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO

Final classification: spill / improper_stacking / safe
```

---

### spill_frame_0031_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.8s | TTFT: 2.17s | 27 tokens @ 40.5 tok/s

**Model response:**
```
Question 1: YES
Question 2: YES

Final classification: spill / improper_stacking / unsafe
```

---

### stacking_frame_0009_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.4s | TTFT: 2.14s | 51 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 50%
```

---

### spill_frame_0168.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 4.2s | TTFT: 2.06s | 85 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Final classification: safe
```

---

### stacking_frame_0026.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 62.3s | TTFT: 2.07s | 2151 tokens @ 35.8 tok/s

**Model response:**
```

1. NO
2. NO
3. NO
4. NO
5. NO
6. NO
7. NO
8. NO
9. NO
10. NO
11. NO
12. NO
13. NO
14. NO
15. NO
16. NO
17. NO
18. NO
19. NO
20. NO
21. NO
22. NO
23. NO
24. NO
25. NO
26. NO
27. NO
28. NO
29. NO
30. NO
31. NO
32. NO
33. NO
34. NO
35. NO
36. NO
37. NO
38. NO
39. NO
40. NO
41. NO
42. NO
43. NO
44. NO
45. NO
46. NO
47. NO
48. NO
49. NO
50. NO
51. NO
52. NO
53. NO
54. NO
55. NO
56. NO
57. NO
58. NO
59. NO
60. NO
61. NO
62. NO
63. NO
64. NO
65. NO
66. NO
67. NO
68. NO
69. NO
70. NO
71. NO
72. NO
73. NO
74. NO
75. NO
76. NO
77. NO
78. NO
79. NO
80. NO
81. NO
82. NO
83. NO
84. NO
85. NO
86. NO
87. NO
88. NO
89. NO
90. NO
91. NO
92. NO
93. NO
94. NO
95. NO
96. NO
97. NO
98. NO
99. NO
100. NO
101. NO
102. NO
103. NO
104. NO
105. NO
106. NO
107. NO
108. NO
109. NO
110. NO
111. NO
112. NO
113. NO
114. NO
115. NO
116. NO
117. NO
118. NO
119. NO
120. NO
121. NO
122. NO
123. NO
124. NO
125. NO
126. NO
127. NO
128. NO
129. NO
130. NO
131. NO
132. NO
133. NO
134. NO
135. NO
136. NO
137. NO
138. NO
139. NO
140. NO
141. NO
142. NO
143. NO
144. NO
145. NO
146. NO
147. NO
148. NO
149. NO
150. NO
151. NO
152. NO
153. NO
154. NO
155. NO
156. NO
157. NO
158. NO
159. NO
160. NO
161. NO
162. NO
163. NO
164. NO
165. NO
166. NO
167. NO
168. NO
169. NO
170. NO
171. NO
172. NO
173. NO
174. NO
175. NO
176. NO
177. NO
178. NO
179. NO
180. NO
181. NO
182. NO
183. NO
184. NO
185. NO
186. NO
187. NO
188. NO
189. NO
190. NO
191. NO
192. NO
193. NO
194. NO
195. NO
196. NO
197. NO
198. NO
199. NO
200. NO
201. NO
202. NO
203. NO
204. NO
205. NO
206. NO
207. NO
208. NO
209. NO
210. NO
211. NO
212. NO
213. NO
214. NO
215. NO
216. NO
217. NO
218. NO
219. NO
220. NO
221. NO
222. NO
223. NO
224. NO
225. NO
226. NO
227. NO
228. NO
229. NO
230. NO
231. NO
232. NO
233. NO
234. NO
235. NO
236. NO
237. NO
238. NO
239. NO
240. NO
241. NO
242. NO
243. NO
244. NO
245. NO
246. NO
247. NO
248. NO
249. NO
250. NO
251. NO
252. NO
253. NO
254. NO
255. NO
256. NO
257. NO
258. NO
259. NO
260. NO
261. NO
262. NO
263. NO
264. NO
265. NO
266. NO
267. NO
268. NO
269. NO
270. NO
271. NO
272. NO
273. NO
274. NO
275. NO
276. NO
277. NO
279. NO
280. NO
281. NO
282. NO
283. NO
284. NO
285. NO
286. NO
287. NO
288. NO
289. NO
290. NO
291. NO
292. NO
293. NO
294. NO
295. NO
296. NO
297. NO
298. NO
299. NO
300. NO
301. NO
302. NO
303. NO
304. NO
305. NO
306. NO
307. NO
308. NO
309. NO
310. NO
310. NO
311. NO
312. NO
313. NO
314. NO
315. NO
316. NO
317. NO
318. NO
319. NO
320. NO
321. NO
322. NO
323. NO
324. NO
325. NO
326. NO
327. NO
328. NO
329. NO
339. NO
340. NO
3410. NO
341. NO
342. NO
343. NO
344. NO
345. NO
346. NO
347. NO
348. NO
349. NO
349. NO
350. NO
340. NO
351. NO
3410. NO
3112. NO
3134. NO
314. NO
315. NO
316. NO
317. NO
318. NO
319. NO
320. NO
313. NO
321. NO
322. NO
323336. NO
327. NO
328. NO
32910. NO
32910. NO
3210. NO
32910. NO
322. NO
320. NO
333333410. NO
32910. NO
33110. NO
320. NO
32110. NO
```

---

### spill_frame_0035_v1.jpg

- **Ground truth:** spill
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.9s | TTFT: 2.06s | 113 tokens @ 39.1 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Then give your final classification:
category: wet_flooring / unsafe_stacking / unsafe
confidence: 0-100%
```

---

### safe_frame_0025.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.2s | TTFT: 2.19s | 40 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1:
Answer: NO

QUESTION 2:
Answer: NO

Final classification:
category: safe
confidence: 100%
```

---

### stacking_frame_0002_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 3.3s | TTFT: 2.3s | 35 tokens @ 40.1 tok/s

**Model response:**
```

Question 1:
Answer: YES

Question 2:
Answer: NO

Final classification: spill / improper_stacking / safe
```

---

### safe_frame_0020.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 60%
- **Time:** 3.1s | TTFT: 2.1s | 38 tokens @ 39.9 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / safe
confidence: 60%
```

---

### safe_frame_0152.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 6.8s | TTFT: 2.12s | 178 tokens @ 38.7 tok/s

**Model response:**
```
QUESTION 1: YES
QUESTION 2: NO

QUESTION 1: The image shows a warehouse with a concrete floor that has visible wet patches, likely due to a spill or a leak.

QUESTION 2: The boxes on the shelves are stacked properly and appear to be in order. There are no signs of the boxes being tilted, hanging off edges, crushed, torn, jumbled, or disorganized.

QUESTION 1: The confidence level for the spill or improper_stacking is 100% due to the visible wet patches on the floor.

QUESTION 2: The confidence level for safe storage is 90% as the boxes appear to be stacked properly and in order, with no signs of danger.
```

---

### spill_frame_0063.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.1s | TTFT: 2.11s | 23 tokens @ 38.4 tok/s

**Model response:**
```

Category: spill / improper_stacking / safe

Confidence: 50%
```

---

### spill_frame_0144.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 7.2s | TTFT: 4.09s | 111 tokens @ 36.9 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: Yes

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: Yes

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%
```

---

### spill_frame_0023_v1.jpg

- **Ground truth:** spill
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 4.2s | TTFT: 2.11s | 79 tokens @ 38.9 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO
```

---

### spill_frame_0034_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 3.1s | TTFT: 2.11s | 38 tokens @ 39.9 tok/s

**Model response:**
```

Question 1: Yes

Question 2: Yes

Final classification: spill / improper_stacking / unsafe

Confidence: 90%
```

---

### stacking_frame_0081.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 2.9s | TTFT: 2.12s | 32 tokens @ 40.1 tok/s

**Model response:**
```

Question 1: Yes

Question 2: Yes

Final Classification: unsafe / spill

Confidence: 90%
```

---

### spill_frame_0150.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.8s | TTFT: 2.12s | 29 tokens @ 40.2 tok/s

**Model response:**
```

Question 1:
Answer: YES

Question 2:
Answer: NO

Final Classification: spill / safe
```

---

### stacking_frame_0005_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 4.2s | TTFT: 2.11s | 79 tokens @ 39.2 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES
```

---

### safe_frame_0018.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 75%
- **Time:** 3.1s | TTFT: 2.13s | 38 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / unsafe
confidence: 75%
```

---

### safe_frame_0097.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.3s | TTFT: 2.13s | 8 tokens @ 44.8 tok/s

**Model response:**
```

QUESTION 1: No
```

---

### spill_frame_0070.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.0s | TTFT: 2.12s | 36 tokens @ 39.9 tok/s

**Model response:**
```
QUESTION 1: Yes
QUESTION 2: Yes

category: spill / improper_stacking / unsafe
confidence: 70%
```

---

### stacking_frame_0079.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.8s | TTFT: 2.13s | 27 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO

category: safe
```

---

### stacking_frame_0006_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.34s | 29 tokens @ 40.5 tok/s

**Model response:**
```

Question 1: No

Question 2: Yes

Final classification: spill / improper_stacking / safe
```

---

### stacking_frame_0028.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 3.1s | TTFT: 2.21s | 38 tokens @ 39.4 tok/s

**Model response:**
```

QUESTION 1: NO

QUESTION 2: YES

category: unsafe / spill / improper_stacking
confidence: 80%
```

---

### stacking_frame_0005_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.1s | TTFT: 2.13s | 37 tokens @ 39.5 tok/s

**Model response:**
```
QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / unsafe
confidence: 70%
```

---

### safe_frame_0167.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.8s | TTFT: 2.16s | 28 tokens @ 40.5 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO

Final Classification: safe
```

---

### safe_frame_0121.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.0s | TTFT: 2.11s | 37 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1: NO

QUESTION 2: NO

Then give your final classification:
category: safe
confidence: 100%
```

---

### stacking_frame_0013_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 70%
- **Time:** 4.9s | TTFT: 2.12s | 108 tokens @ 39.2 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 70%
```

---

### spill_frame_0085.jpg

- **Ground truth:** spill
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.12s | 26 tokens @ 40.2 tok/s

**Model response:**
```

Question 1:
Answer: YES

Question 2:
Answer: NO

Final classification: safe
```

---

### safe_frame_0147.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 3.2s | TTFT: 2.15s | 43 tokens @ 39.4 tok/s

**Model response:**
```

Question 1:
Answer: NO

Question 2:
Answer: YES

Final classification: spill / improper_stacking / safe
Confidence: 85%
```

---

### safe_frame_0192.jpg

- **Ground truth:** safe
- **Predicted:** improper_stacking
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 4.4s | TTFT: 2.18s | 90 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1: Yes. There is a wet spot on the floor near the stacked boxes.

QUESTION 2: Yes. The boxes on the shelves are not properly stacked and some of them appear to be leaning or falling over.

FINAL CLASSIFICATION: spill / improper_stacking / unsafe

CONFIDENCE: 80%
```

---

### safe_frame_0141.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.6s | TTFT: 2.29s | 20 tokens @ 40.7 tok/s

**Model response:**
```

Question 1: NO

Question 2: NO

Final classification: safe
```

---

### stacking_frame_0096.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.1s | TTFT: 2.12s | 39 tokens @ 39.9 tok/s

**Model response:**
```

QUESTION 1: YES

QUESTION 2: NO


category: spill / improper_stacking / safe
confidence: 50%
```

---

### safe_frame_0069.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.1s | TTFT: 2.12s | 40 tokens @ 39.6 tok/s

**Model response:**
```

QUESTION 1: Yes.

QUESTION 2: Yes.

category: spill / improper_stacking / unsafe
confidence: 70%
```

---

### safe_frame_0059.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.13s | 28 tokens @ 38.9 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO

Final classification: safe
```

---

### stacking_frame_0007_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.7s | TTFT: 2.21s | 22 tokens @ 40.5 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES
```

---

### spill_frame_0124.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 85%
- **Time:** 3.1s | TTFT: 2.11s | 37 tokens @ 38.5 tok/s

**Model response:**
```

Answer 1: YES

Answer 2: YES

Final classification: spill / improper_stacking / unsafe
Confidence: 85%
```

---

### stacking_frame_0087.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.6s | TTFT: 2.17s | 14 tokens @ 40.5 tok/s

**Model response:**
```

Category: safe
Confidence: 100%
```

---

### spill_frame_0008.jpg

- **Ground truth:** spill
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.6s | TTFT: 2.22s | 16 tokens @ 41.5 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes
```

---

### safe_frame_0074.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 85%
- **Time:** 3.1s | TTFT: 2.14s | 38 tokens @ 39.8 tok/s

**Model response:**
```

Answer 1: YES

Answer 2: YES

Final classification: spill / improper_stacking / unsafe

Confidence: 85%
```

---

### safe_frame_0178.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.7s | TTFT: 2.18s | 22 tokens @ 40.4 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO
```

---

### stacking_frame_0041.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 2.9s | TTFT: 2.12s | 30 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1: No

QUESTION 2: No

category: safe
confidence: 100%
```

---

### stacking_frame_0018_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 75%
- **Time:** 2.9s | TTFT: 2.38s | 27 tokens @ 40.3 tok/s

**Model response:**
```
QUESTION 1: YES
QUESTION 2: YES

category: unsafe
confidence: 75%
```

---

### spill_frame_0104.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 50%
- **Time:** 2.7s | TTFT: 2.15s | 22 tokens @ 40.4 tok/s

**Model response:**
```

Category: spill / improper_stacking / safe
Confidence: 50%
```

---

### safe_frame_0123.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 80%
- **Time:** 3.4s | TTFT: 2.12s | 49 tokens @ 39.6 tok/s

**Model response:**
```

Answer: 

QUESTION 1: YES

QUESTION 2: YES

Then give your final classification:
category: unsafe_storage / potential_hazard
confidence: 80%
```

---

### stacking_frame_0099.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.13s | 28 tokens @ 40.0 tok/s

**Model response:**
```

QUESTION 1:
Answer: NO

QUESTION 2:
Answer: NO

Final classification: safe
```

---

### safe_frame_0053.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.6s | TTFT: 2.19s | 16 tokens @ 41.4 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes
```

---

### stacking_frame_0100.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.4s | TTFT: 2.15s | 50 tokens @ 39.2 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO

FINAL CLASSIFICATION:

category: safe
confidence: 70-100%
```

---

### safe_frame_0090.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 4.1s | TTFT: 2.11s | 79 tokens @ 38.8 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO
```

---

### spill_frame_0134.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 5.0s | TTFT: 2.23s | 108 tokens @ 38.8 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 90%
```

---

### safe_frame_0056.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.8s | TTFT: 2.15s | 26 tokens @ 40.6 tok/s

**Model response:**
```

Question 1:
Answer: NO

Question 2:
Answer: NO

Final classification: safe
```

---

### spill_frame_0033_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 5.1s | TTFT: 2.22s | 111 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%
```

---

### spill_frame_0112.jpg

- **Ground truth:** spill
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.5s | TTFT: 2.13s | 14 tokens @ 41.4 tok/s

**Model response:**
```
QUESTION 1: YES
QUESTION 2: YES
```

---

### spill_frame_0033_v2.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 3.3s | TTFT: 2.35s | 38 tokens @ 39.9 tok/s

**Model response:**
```

Question 1: YES

Question 2: YES

Final classification: spill / improper_stacking / unsafe

Confidence: 90%
```

---

### spill_frame_0167.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 3.1s | TTFT: 2.23s | 37 tokens @ 39.6 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES

Final classification: spill / improper_stacking / unsafe
```

---

### safe_frame_0099.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 90%
- **Time:** 2.9s | TTFT: 2.14s | 29 tokens @ 40.2 tok/s

**Model response:**
```

QUESTION 1: NO

QUESTION 2: YES

category: unsafe
confidence: 90%
```

---

### stacking_frame_0063.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 4.1s | TTFT: 2.16s | 79 tokens @ 39.3 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES
```

---

### safe_frame_0102.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.1s | TTFT: 2.11s | 38 tokens @ 39.9 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / unsafe
confidence: 50%
```

---

### spill_frame_0140.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.3s | TTFT: 2.11s | 46 tokens @ 39.6 tok/s

**Model response:**
```
QUESTION 1: YES
QUESTION 2: NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%
```

---

### stacking_frame_0097.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.5s | TTFT: 2.13s | 16 tokens @ 41.3 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes
```

---

### safe_frame_0062.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.8s | TTFT: 2.19s | 26 tokens @ 40.3 tok/s

**Model response:**
```

Question 1:
Answer: YES

Question 2:
Answer: YES

Final classification: unsafe
```

---

### stacking_frame_0061.jpg

- **Ground truth:** improper_stacking
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.7s | TTFT: 2.12s | 22 tokens @ 40.8 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES
```

---

### safe_frame_0071.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.6s | TTFT: 2.33s | 18 tokens @ 40.6 tok/s

**Model response:**
```
Question 1: NO
Question 2: NO

Final Classification: safe
```

---

### safe_frame_0049.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.19s | 28 tokens @ 40.4 tok/s

**Model response:**
```

QUESTION 1:
Answer: NO

QUESTION 2:
Answer: NO

Final classification: safe
```

---

### safe_frame_0011.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 80%
- **Time:** 3.7s | TTFT: 2.14s | 54 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO

FINAL CLASSIFICATION:
category: spill / improper_stacking / safe
confidence: 80%
```

---

### spill_frame_0033_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.4s | TTFT: 2.16s | 51 tokens @ 39.4 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 70%
```

---

### spill_frame_0007.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.4s | TTFT: 2.13s | 50 tokens @ 39.5 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES

CATEGORY: spill / improper_stacking / unsafe
CONFIDENCE: 70%
```

---

### stacking_frame_0059.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.5s | TTFT: 2.12s | 54 tokens @ 39.1 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: NO

FINAL CLASSIFICATION:
category: spill / improper_stacking / safe
confidence: 70%
```

---

### spill_frame_0022_v1.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 80%
- **Time:** 4.9s | TTFT: 2.29s | 103 tokens @ 38.7 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: YES

Final Classification: spill / improper_stacking / unsafe

Confidence: 80%
```

---

### stacking_frame_0051.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.11s | 31 tokens @ 40.3 tok/s

**Model response:**
```

QUESTION 1:
Answer: NO

QUESTION 2:
Answer: NO

Final classification:
category: safe
```

---

### safe_frame_0098.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 3.9s | TTFT: 2.13s | 67 tokens @ 39.4 tok/s

**Model response:**
```
QUESTION 1: No
QUESTION 2: Yes

The floor of the warehouse appears to be wet and slippery. The boxes on the shelves are stacked improperly, creating a risk of items falling off the edges. The overall condition of the warehouse is unsafe.
```

---

### stacking_frame_0031.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.0s | TTFT: 2.12s | 34 tokens @ 39.2 tok/s

**Model response:**
```

1. NO
2. YES

category: spill / improper_stacking / safe
confidence: 0-100%
```

---

### stacking_frame_0020_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.3s | TTFT: 2.29s | 37 tokens @ 39.8 tok/s

**Model response:**
```

Category: Safe
Confidence: 100%

Explanation: The warehouse appears to be well-organized and safe for workers.
```

---

### safe_frame_0041.jpg

- **Ground truth:** safe
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.6s | TTFT: 2.15s | 14 tokens @ 41.9 tok/s

**Model response:**
```
QUESTION 1: NO
QUESTION 2: NO
```

---

### safe_frame_0160.jpg

- **Ground truth:** safe
- **Predicted:** safe
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.6s | TTFT: 2.15s | 20 tokens @ 40.7 tok/s

**Model response:**
```

Question 1: NO

Question 2: NO

Final classification: safe
```

---

### safe_frame_0172.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 75%
- **Time:** 3.1s | TTFT: 2.11s | 38 tokens @ 39.8 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: unsafe / improper_stacking / spill
confidence: 75%
```

---

### safe_frame_0176.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.4s | TTFT: 2.14s | 51 tokens @ 39.4 tok/s

**Model response:**
```

QUESTION 1:
Answer: NO

QUESTION 2:
Answer: YES

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 70%
```

---

### spill_frame_0127.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 100%
- **Time:** 5.0s | TTFT: 2.12s | 111 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: YES

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Then give your final classification:
category: spill / improper_stacking / safe
confidence: 0-100%
```

---

### spill_frame_0149.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 70%
- **Time:** 3.4s | TTFT: 2.12s | 50 tokens @ 39.3 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES

CATEGORY: spill / improper_stacking / safe
CONFIDENCE: 70%
```

---

### stacking_frame_0019_v1.jpg

- **Ground truth:** improper_stacking
- **Predicted:** safe
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 4.7s | TTFT: 2.2s | 100 tokens @ 39.0 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: NO

QUESTION 2: Are there any boxes on the shelves that are tilted, hanging off edges, crushed, torn, jumbled, disorganized, or look like they could fall?
Answer: NO

Then give your final classification:
category: safe
confidence: 100%
```

---

### stacking_frame_0006_v0.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 90%
- **Time:** 3.1s | TTFT: 2.12s | 38 tokens @ 39.2 tok/s

**Model response:**
```

QUESTION 1: Yes

QUESTION 2: Yes

category: spill / improper_stacking / unsafe
confidence: 90%
```

---

### stacking_frame_0073.jpg

- **Ground truth:** improper_stacking
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 100%
- **Time:** 3.0s | TTFT: 2.12s | 37 tokens @ 39.9 tok/s

**Model response:**
```
Question 1: YES
Question 2: YES

category: spill / improper_stacking / unsafe
confidence: 0-100%
```

---

### spill_frame_0038_v0.jpg

- **Ground truth:** spill
- **Predicted:** spill
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.1s | TTFT: 2.14s | 34 tokens @ 40.0 tok/s

**Model response:**
```
Question 1: Yes
Question 2: Yes

category: spill / improper_stacking / unsafe
confidence: 50%
```

---

### spill_frame_0158.jpg

- **Ground truth:** spill
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.14s | 29 tokens @ 39.8 tok/s

**Model response:**
```

QUESTION 1: Is there any liquid, wet patch, or puddle visible on the floor?
Answer: Yes
```

---

### spill_frame_0026_v1.jpg

- **Ground truth:** spill
- **Predicted:** None
- **Result:** WRONG
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.7s | TTFT: 2.11s | 22 tokens @ 40.6 tok/s

**Model response:**
```

QUESTION 1:
Answer: YES

QUESTION 2:
Answer: YES
```

---

### stacking_frame_0037.jpg

- **Ground truth:** improper_stacking
- **Predicted:** improper_stacking
- **Result:** CORRECT
- **Severity:** None
- **Confidence:** None%
- **Time:** 2.9s | TTFT: 2.14s | 27 tokens @ 40.2 tok/s

**Model response:**
```

QUESTION 1: No

QUESTION 2: Yes

category: unsafe / improper_stacking
```

---

### safe_frame_0068.jpg

- **Ground truth:** safe
- **Predicted:** spill
- **Result:** WRONG
- **Severity:** None
- **Confidence:** 50%
- **Time:** 3.1s | TTFT: 2.24s | 37 tokens @ 39.6 tok/s

**Model response:**
```

Question 1: Yes

Question 2: Yes

Final classification: spill / improper_stacking / unsafe
Confidence: 50%
```

---

