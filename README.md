# Matlab 筆記

### 解決多筆數據index排序問題
**Note:** 重複收集實驗資料51次，每次有1152筆資料，但是每一次的index都是亂的。
```matlab
ds = tabularTextDatastore('largedump.txt'); % largedump.txt is messy data.
largedump = table2array(readall(ds)); % Convert talbe to array.

cutnum = repmat(1152,1,51); % Experiment repeat 51 times, each times has 1152 data.
largedumpc = mat2cell(largedump,cutnum); % Convert array to cell array, each cell has 1152 data.
largedumpr = cell2mat(cellfun(@sortrows,largedumpc,'Un',0));  % Sort the data in the cell.
```
### 批次建立legend和改變legend的排列
**Note:** 建立X1、X2、X3、X4的legend並排成2排，不使用compose的話就變成```legend('X1','X2','X3','X4')```，當需要標註的數量變多時就會造成困擾。
```matlab 
t=0:0.01:2*pi;
x1 = sin(t);
x2 = cos(t);
x3 = t+0.2;
x4 = -t+sin(t);
plot(t,x1,t,x2,t,x3,t,x4); axis tight
index = 1:4;
lgd = legend(compose("X%d",index)) % Create lengend
lgd.NumColumns = 2;
```
