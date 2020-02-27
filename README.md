# Matlab 筆記

## 目錄
* [標題](#標題Legend)

### 標題Legend


### 刪除記憶體變數
```matlab
clearvars a b c
```
### 矩陣分解
**Note:** mat2cell第二個變數是要拆的row，第三個變數是要拆的column
```matlab
A = reshape(1:20,5,4);
B = mat2cell(A,[3 2],[2 2]);
C = mat2cell(A,[4 1]);  % Ignore variable 3
D = mat2cell(A,5,ones(1,4)); % Split into one by one column
```
### 比較每個元素對某向量的關係
```matlab
raw_data = randi(10,3); % 隨機產生1到10的3*3矩陣
vector = 1:10;
data = arrayfun(@(x) gt(x,vector),raw_data,'Un',0); % ('Un',0) means : returns the outputs in cell arrays.
```
### import大型文字檔，選取部分檔案範圍
**Note:** ds.SelectedVariableNames可以選擇要的檔案範圍
```matlab
ds = tabularTextDatastore('filename.txt');
ds.SelectedVariableNames([9:11 13:end]) = []; % Only need 1~8 and 12 columns
data = readall(ds); % Read all data in datastore
data = table2array(data); % Convert table to array
```
### 解決多筆數據index排序問題
**Note:** 重複收集實驗資料51次，每次有1152筆資料，但是每一次的index都是亂的。
```matlab
ds = tabularTextDatastore('largedump.txt'); % largedump.txt is messy data.
largedump = table2array(readall(ds)); % Convert talbe to array.

cutnum = repmat(1152,1,51); % Experiment repeat 51 times, each times has 1152 data.
largedumpc = mat2cell(largedump,cutnum); % Convert array to cell array, each cell has 1152 data.
largedumpr = cell2mat(cellfun(@sortrows,largedumpc,'Un',0));  % Sort the data in the cell.
```
### Convert symbolic expression to function handle
```matlab 
syms x y
f(x,y) = x^3 + y^3;
ht = matlabFunction(f)
```
### 提取symbolic expression的係數
```matlab
syms x
c = sym2poly(x^3 - 2*x - 5)
```
### 多變量函數取係數
```matlab
syms x y
cx = coeffs(3*x*y+2*y,y)
cxy = coeffs(3*x*y+2*y,[x y])
```
### 在figure中建立子圖
```matlab
t = 0:0.01:20;
x1 = sin(t);
x2 = t;
x3 = t.^2;
plot(t,[x1;x2])
axes('position',[.20 .65 .3 .25])
box on % put box around new pair of axes
plot(t,x3)Multivariate 
```
### 內插
```matlab
t = 0:0.3:2*pi; 
x = sin(t);
plot(t,x,'o');axis tight;hold on
tq = 0:0.1:2*pi;
xq = interp1(t,x,tq);
plot(tq,xq,'.');
```
### ode45用於矩陣微分方程
```matlab
A = [1 2;3 4];
t = 0:0.05:5;
x0 = [1 2;3 -1];
[Tx,X] = ode45(@(t,x) xDRE(t,x,A),t,x0);
plot(Tx,X);

function dxdt = xDRE(~,x,A)
X = reshape(x,size(A));  % Reshape input p into matrix
xdot = -X*A-A;
dxdt = xdot(:);  % Reshape output as a column vector
end
```
### fig檔存檔
```matlab
savefig(gcf,'test.fig')
```
### 顯示matlab所有設定
```matlab
s = get(groot,'factory');
```
**Note:** 例如s.factoryAxesFontName是Helvetica，要改成Times New Roman就要
```set(groot,'DefaultAxesFontName','Times New Roman')```或是```set(0,'DefaultAxesFontName','Times New Roman')```

---
## Legend Settings
### 批次建立legend和改變legend的排列
**Note:** 建立X1、X2、X3、Z1的legend，不使用compose的話就變成```legend('X1','X2','X3','Z1')```，當需要標註的數量變多時就會造成困擾。
```matlab 
t = 0:0.01:2*pi;
x1 = sin(t);
x2 = cos(t);
x3 = t+0.2;
z1 = -t+sin(t);
figure,plot(t,x1,t,x2,t,x3,t,z1); axis tight
index = 1:3;
lgd = legend([compose("X%d",index) 'Z1']); % Create lengend
lgd.NumColumns = 2; % Number of columns
```
### 改變所有legend的字體大小
```matlab 
set(findall(0,'Type','legend'),'FontSize',11) % Set all legends font size
```
### 移除legend背景顏色
```matlab
t = 0:0.01:2*pi;
x = t.^2;
plot(t,x); axis tight
legend('x','color','none') % Remove Legend background color
```
---
## 設定Figure
### 移除emf檔圖片背景顏色
```matlab
set(0,'DefaultFigureInvertHardcopy','off');
set(gcf,'color','none'); % Graphic background is set to colorless
set(gca,'color','none'); % Axis background set to colorless
```
### 更改預設設定
```matlab
set(0,'DefaultAxesFontName','Times New Roman')  % Set all font style
set(0,'DefaultAxesFontWeight','bold') % Set x y axis font style
% set(0,'DefaultAxesFontSize',11) % Set all figure font size
set(0,'DefaultAxesLabelFontSizeMultiplier',1.2) % Set x y label size
set(0,'DefaultAxesTitleFontSizeMultiplier',1.2) % Set all title size
% set(0,'DefaultAxesTitleFontWeight','normal')  % Set all title style
set(0,'DefaultAxesLooseInset',[0.0774 0.1016 0.0134 0.0540]) % Delete figure White Space % data from get(gca,'TightInset') 
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on') % Show x and y grid
% set(0,'DefaultFigurePaperPositionMode','manual')
% set(0,'DefaultFigurePaperUnits','inches')
% set(0,'DefaultFigurePosition',[700 650 14.4*37.81 14.28*0.618*37.81]) % Set figure window position,size and resolution
% set(0,'DefaultFigureWindowState','maximized') % Set figure fullscreen
% set(0,'DefaultFigureWindowStyle','docked') % Set figure window style
set(0,'DefaultLineLineWidth',1.5) % Set all line width
% set(0,'DefaultFigureInvertHardcopy','off');
```
**Note:** 原始設定
```matlab
set(0,'DefaultAxesFontName','Helvetica')
set(0,'DefaultAxesFontWeight','normal')
% set(0,'DefaultAxesFontSize',10)
set(0,'DefaultAxesLabelFontSizeMultiplier',1.1)
set(0,'DefaultAxesTitleFontSizeMultiplier',1.1)
set(0,'DefaultAxesTitleFontWeight','bold')
set(0,'DefaultAxesLooseInset',[0.13 0.11 0.095 0.075]);
set(0,'DefaultAxesXGrid','off','DefaultAxesYGrid','off')
set(0,'DefaultFigurePaperPositionMode','auto') % Ensure that jpg output is the same as the actual window on screen
% set(0,'DefaultFigurePaperUnits','inches')
set(0,'DefaultFigurePosition',[680 558 560 420])
set(0,'DefaultFigureWindowStyle','normal')
set(0,'DefaultLineLineWidth',0.5)
% set(0,'DefaultFigureInvertHardcopy','on');
```
