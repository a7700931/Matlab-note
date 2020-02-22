# Matlab 筆記

### 解決多筆數據index排序問題
**Note:** 重複收集實驗資料51次，每次有1152筆資料，但是每一次的index都是亂的
```matlab
ds = tabularTextDatastore('largedump.txt'); % largedump.txt is messy data.
largedump = table2array(readall(ds)); % Convert talbe to array.

cutnum = repmat(1152,1,51); % Experiment repeat 51 times, each times has 1152 data.
largedumpc = mat2cell(largedump,cutnum); % Convert array to cell array, each cell has 1152 data.
largedumpr = cell2mat(cellfun(@sortrows,largedumpc,'Un',0));  % Sort the data in the cell.
```
