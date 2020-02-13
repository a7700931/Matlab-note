clearvars
close all
clc
tic
disp('Beginning post-processing');
% Parameters
n_atoms = 4703;     % number of atoms in total
n_base = 4608;      % number of atoms in base
timesteps = 251;    % number of timesteps
xbins = 16;         % number of bins in x/y/z direction
ybins = 8;
zbins = 4;
tempxbins = xbins/2;      % number of bins for temperature calcs.leave it alone
tempybins = 8;
tempzbins = 4;
n_bins = xbins*ybins*zbins+1;  % number of bins
temp_n_bins = tempxbins*tempybins*tempzbins+1;  % number of bins for temp.
contourRes = 2;       % Resolution of contour for potential
contourHeight = 5;      % Height of contour zone above base 

k = 1.38064852*10^(-23);   % Boltzmann constant

dump_file = 'largedump.txt';
position_file = 'LargerInput.txt';
% prefix for output, filename will be 'output_file'Results-000.csv and
% 'output_dir''output_file'Probe-000.csv, with the numbers replaced by the time step.
output_dir = 'TriboResults/';
output_file_prefix = 'LargeInput';

% Read the atom list, bin the atoms
disp('Reading initial atom positions');


% atom_file = fopen(position_file, 'r');
% format_spec = '%d %d %d %f %f %f';
% array_size = [6,Inf];
% atoms = fscanf(atom_file, format_spec, array_size); % read file.
% atoms = atoms(4:6, :);
% atoms = atoms';
atoms = load(position_file);
atoms(:,1:3)=[];

% Parameters for the bounds of the SiO2 lattice. See the LAMMPS input file
% for reference.
nx = xbins;
ny = ybins;
nz = zbins;

bxo = 4.913400;
byo = 4.255129;
bzo = 5.405200;
xyo = -2.456700;

xlo = 0;
xhi = nx*bxo;
ylo = 0;
yhi = ny*byo;
zlo = 0;
zhi = 21; %nz*bzo;
xy = ny*xyo;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Binning Atoms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Creating bins');

% Change the x direction for binning to a tilted direction.
% Does not handle 3D tilt, only xy tilt.
tilted_atoms = zeros(size(atoms));
tilted_atoms(:,1) = (atoms(:,1)-(atoms(:,2)/(yhi-ylo)).*xy)/(xhi-xlo);
tilted_atoms(:,2) = atoms(:,2)/(yhi-ylo);
tilted_atoms(:,3) = atoms(:,3)/(zhi-zlo);

% Label each atom with a bin number. 
atom_bins = zeros(n_atoms, 1);
% bin_coords = zeros(size(atoms));

% bin_coords(:,1) = floor(xbins*tilted_atoms(:,1));
% bin_coords(:,2) = floor(ybins*tilted_atoms(:,2));
% bin_coords(:,3) = floor(zbins*tilted_atoms(:,3));
bin_coords = floor([xbins ybins zbins].*tilted_atoms);

atom_bins(:,1) = 1 + bin_coords(:,1) + bin_coords(:,2)*xbins + bin_coords(:,3)*(xbins*ybins);
atom_bins(atom_bins>n_bins) = n_bins;   % bin for any unbinned atoms, or the target group.

bin_atoms = zeros(n_bins, n_base/(n_bins-1));
bin_count = zeros(n_bins, 1);

for i=1:n_atoms
    bin_atoms(atom_bins(i), bin_count(atom_bins(i))+1) = i;
    bin_count(atom_bins(i)) = bin_count(atom_bins(i)) + 1;
end

% Creating temperature bins

% Label each atom with a bin number. 
temp_atom_bins = zeros(n_atoms, 1);
% temp_bin_coords = zeros(size(atoms));

% temp_bin_coords(:,1) = floor(tempxbins*tilted_atoms(:,1));
% temp_bin_coords(:,2) = floor(tempybins*tilted_atoms(:,2));
% temp_bin_coords(:,3) = floor(tempzbins*tilted_atoms(:,3));
temp_bin_coords = floor([tempxbins tempybins tempzbins].*tilted_atoms);

temp_atom_bins(:,1) = 1 + temp_bin_coords(:,1) + temp_bin_coords(:,2)*tempxbins + temp_bin_coords(:,3)*(tempxbins*tempybins);
temp_atom_bins(temp_atom_bins>temp_n_bins) = temp_n_bins;   % bin for any unbinned atoms, or the target group.

temp_bin_atoms = zeros(temp_n_bins, n_base/(temp_n_bins-1));
temp_bin_count = zeros(temp_n_bins, 1);

for i=1:n_atoms
    temp_bin_atoms(temp_atom_bins(i), temp_bin_count(temp_atom_bins(i))+1) = i;
    temp_bin_count(temp_atom_bins(i)) = temp_bin_count(temp_atom_bins(i)) + 1;
end

% Read the dump file. The lines that aren't data need to be deleted in
% advance.
disp('Reading dump file');

% fileID = fopen(dump_file, 'r');
% formatSpec = '%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f';
% sizeDump = [21, Inf];
% atom_dump = fscanf(fileID, formatSpec, sizeDump);
% fclose(fileID);

% atom_dump = atom_dump([1:8 12], :);

% atom_dump = atom_dump';
ds = tabularTextDatastore(dump_file);
ds.SelectedVariableNames([9:11 13:end]) = [];
T = readall(ds);
atom_dump = table2array(T);

% Split the dump into discrete time steps.
atom_data = zeros(n_atoms, 9, timesteps);
for t=1:timesteps
    raw_atom_data = atom_dump((1+(t-1)*n_atoms):(t*n_atoms), :);
    atom_data(:,1:9,t) = sortrows(raw_atom_data);
end

% Remove bins from the simulation if they are on the edge. This deals with
% the weirdness resulting from periodic boundary conditions, which causes
% the simulation to teleport atoms across the box.
disp('Trimming simulation domain');
keep_atoms = ((bin_coords(:,1)~=0) & (bin_coords(:,1)~=(xbins-1)) & ...
    (bin_coords(:,2)~=0) & (bin_coords(:,2)~=(ybins-1)));

atom_data = atom_data(keep_atoms,:,:);

% Take the data and divide it into individual vectors.
atom_id(:,:) = atom_data(:,1,:);
type(:,:) = atom_data(:,2,:);
x(:,:) = atom_data(:,3,:);
y(:,:) = atom_data(:,4,:);
z(:,:) = atom_data(:,5,:);
vx(:,:) = atom_data(:,6,:);
vy(:,:) = atom_data(:,7,:);
vz(:,:) = atom_data(:,8,:);
q(:,:) = atom_data(:,9,:);

% Calculate which bin each atom is in.
disp('Calculating bins');
bins = atom_bins(atom_id(:,:));
tempbins = temp_atom_bins(atom_id(:,:));

% Calculate the kinetic temperature of each atom.
disp('Calculating temperatures');
mass = (28.0855*(type==1) + 15.9994*(type==2) + 12.0107*(type==3)) / (6.022*10^26);  % mass, kg
velocity = 100 * sqrt(vx.^2 + vy.^2 + vz.^2);     % velocity magnitude, m/s
kintemp = (mass.*(velocity.^2)) ./ (3*k);    % kinetic temperature

% Calculate the temperature for each bin.
temp = zeros(temp_n_bins,timesteps);
% for i=1:temp_n_bins
%     for t=1:timesteps
%         temp(i,t) = mean(kintemp(tempbins(:,t)==i,t));
%     end
% end
for i=1:temp_n_bins
    tempbinsTF = tempbins==i;
    kintempr = reshape(kintemp(tempbinsTF),[],timesteps);
    temp(i,:) = mean(kintempr);
end

bintemp = temp(tempbins(:,1),:);        % bin average temperature

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Polarization
%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Calculating polarization');

% Find the volumetric center of each bin.
centx = zeros(n_bins, timesteps);
centy = zeros(n_bins, timesteps);
centz = zeros(n_bins, timesteps);
% for i=1:n_bins
%     for t=1:timesteps
%         centx(i,t) = mean(x(bins(:,t)==i,t));
%         centy(i,t) = mean(y(bins(:,t)==i,t));
%         centz(i,t) = mean(z(bins(:,t)==i,t));
%     end
% end

% for i=1:n_bins
%     binTF = sparse(bins==i);
%     xbr = reshape(x(binTF),[],timesteps);
%     ybr = reshape(y(binTF),[],timesteps);
%     zbr = reshape(z(binTF),[],timesteps);
%     centx(i,:) = mean(xbr);
%     centy(i,:) = mean(ybr);
%     centz(i,:) = mean(zbr);
% end

for i=1:n_bins
    binTF = sparse(bins(:,1)==i);
    centx(i,:) = mean(x(binTF,:));
    centy(i,:) = mean(y(binTF,:));
    centz(i,:) = mean(z(binTF,:));
end

% Find the distance of each atom from the center of the bin.
% distx = zeros(size(atom_data,1), timesteps);
% disty = zeros(size(atom_data,1), timesteps);
% distz = zeros(size(atom_data,1), timesteps);
% parfor i=1:size(atom_data,1)
%     distx(i,:) = x(i,:) - centx(bins(i,1),:);
%     disty(i,:) = y(i,:) - centy(bins(i,1),:);
%     distz(i,:) = z(i,:) - centz(bins(i,1),:);
% end
binsd = bins(:,1);

centxd = centx(binsd,:);
centyd = centy(binsd,:);
centzd = centz(binsd,:);

distx = x - centxd;
disty = y - centyd;
distz = z - centzd;

% Find the polarization for each atom in x,y,z, respectively.
pol_x = q.*distx;
pol_y = q.*disty;
pol_z = q.*distz;

% Calculate the polarization in the x, y, and z directions for each bin.
bin_dipole = zeros(n_bins, 3, timesteps);
% for i=1:n_bins
%     for t=1:timesteps
%         bin_dipole(i,1,t) = sum(pol_x(bins(:,t)==i,t));
%         bin_dipole(i,2,t) = sum(pol_y(bins(:,t)==i,t));
%         bin_dipole(i,3,t) = sum(pol_z(bins(:,t)==i,t));
%     end
% end

% for i=1:n_bins
%     binTF = sparse(bins==i);
%     pol_xr = reshape(pol_x(binTF),[],timesteps);
%     pol_yr = reshape(pol_y(binTF),[],timesteps);
%     pol_zr = reshape(pol_z(binTF),[],timesteps);
%     bin_dipole(i,1,:) = sum(pol_xr);
%     bin_dipole(i,2,:) = sum(pol_yr);
%     bin_dipole(i,3,:) = sum(pol_zr);
% end

for i=1:n_bins
    binTF = sparse(bins(:,1)==i);
    bin_dipole(i,1,:) = sum(pol_x(binTF,:));
    bin_dipole(i,2,:) = sum(pol_y(binTF,:));
    bin_dipole(i,3,:) = sum(pol_z(binTF,:));
end

relBinPol = bin_dipole(:,:,:) - bin_dipole(:,:,4);

xbinpol = zeros(size(atom_data,1), timesteps);
ybinpol = zeros(size(atom_data,1), timesteps);
zbinpol = zeros(size(atom_data,1), timesteps);
% for i=1:size(atom_data,1)
%     xbinpol(i,:) = bin_dipole(bins(i,1),1,:);
%     ybinpol(i,:) = bin_dipole(bins(i,1),2,:);
%     zbinpol(i,:) = bin_dipole(bins(i,1),3,:);
% end

xbinpol(:,:) = bin_dipole(binsd,1,:);
ybinpol(:,:) = bin_dipole(binsd,2,:);
zbinpol(:,:) = bin_dipole(binsd,3,:);

% Calculate the total polarization.
bin_pol = sqrt(xbinpol.^2 + ybinpol.^2 + zbinpol.^2);

% Calculate the relative polarization
pol_start = bin_pol(:,5);
rel_pol(:,:) = bin_pol(:,:) - pol_start;

% Calculate potential
disp('Calculating potentials');
potential = zeros(size(atom_data,1),n_bins,timesteps);
voltage = zeros(size(atom_data,1),timesteps);
elecField = zeros(size(atom_data,1),timesteps,3);

% for i = 1:1%size(atom_data,1)
%     for t = 1:1%timesteps
%         for j = 1:n_bins
%             if j~=bins(i,t) && sum(bins(:,t)==j)>0  % not this atoms bins, and not a bin that's been trimmed.
%                 pos_vec = [x(i,t)-centx(j,t), y(i,t)-centy(j,t), z(i,t)-centz(j,t)];
%                 % Potential
%                 dist_vec = pos_vec/sqrt(sum(pos_vec.^2))^3;
%                 potential(i,j,t) = sqrt(sum((bin_dipole(j,:,t).*dist_vec).^2));
%                 % Electric Field
%                 fieldTensor = 3*(pos_vec * pos_vec') ./ sqrt(sum(pos_vec.^2))^5;
%                 fieldTensor = fieldTensor - 1/sqrt(sum(pos_vec.^2))^3;
%                 elecField(i,t,:) = elecField(i,t,:) + reshape(bin_dipole(j,:,t) * fieldTensor,[1,1,3]);
%             end
%         end
%         voltage(i,t) = sum(potential(i,:,t));
%     end
% end
% test1 = potential(i,:,t);
indexj = 1:n_bins;	%index
bins2c = num2cell(bins);	%bins 轉cell array
loop_c = cellfun(@(x) indexj ~= x,bins2c,'UniformOutput',0);	%條件一:判斷j~=bins(i,t) bins每個元素都要和1到n_bins比較

bins2atom = mat2cell(bins,size(atom_data,1),ones(1,timesteps));    %以column為單位bins拆成cell array
indexEq = cellfun(@(x) indexj == x,bins2atom,'UniformOutput',0);    %判斷bins(:,t)==j
indexS = cellfun(@sum,indexEq,'UniformOutput',0);   %加總判斷結果
indexGT0 = cellfun(@(x) x>0,indexS,'UniformOutput',0);  %條件二:總和大於零

cond = and(cell2mat(loop_c),cell2mat(indexGT0)); %兩個條件為真
indexAll = 1:timesteps*n_bins;
indexAllr = reshape(indexAll,n_bins,[]);

for i = 1:size(atom_data,1)
    for t = 1:timesteps
        condsel = cond(i,indexAllr(:,t));	%選要的條件
        pos_vec = [x(i,t)-centx(:,t) y(i,t)-centy(:,t) z(i,t)-centz(:,t)];
        % Potential
        t1=sum(pos_vec.^2,2);
        den = sqrt(sum(pos_vec.^2,2));
        dist_vec = pos_vec./den.^3;
        dist_vec(~condsel,:) = 0;	%把NaN換成0
        potential(i,:,t) = sqrt(sum((bin_dipole(:,:,t).*dist_vec).^2,2));
        % Electric Field
        fieldTensor = 3*sum(pos_vec.^2,2)./den.^5 - 1./den.^3;
        elecField_element = bin_dipole(:,:,t) .* fieldTensor;
        elecField_element(~condsel,:) = 0;	%把NaN換成0
        elecField(i,t,:) = reshape(sum(elecField_element),1,1,3);
        
        voltage(i,t) = sum(potential(i,:,t),2);
    end
end

% Unit conversion
voltage = voltage ./ (4*pi*8.85*10^(-12)).*(1.60217662*10^(-19)).*(10^10);      % Answer is in Volts.
elecField = elecField ./ (4*pi*8.85*10^(-12)).*(1.60217662*10^(-19)).*(10^10);  % Answer is in V/m
rel_voltage = voltage(:,:) - voltage(:,5);

% Calculate voltage gradient between probe and base.
disp('Calculating contour positions');

% Find area that the contour will be calculated in.
conxlo = min(x(:,1));
conxhi = max(x(:,1));
conylo = min(y(:,1));
conyhi = max(y(:,1));
conzlo = max(z(bins(:,1)~=n_bins)) + contourRes;
conzhi = conzlo + contourHeight;
shift = xy*(conyhi-conylo)/(yhi-ylo);
if xy<0
    conxlo = conxlo-shift;
    conxhi = conxhi+shift;
end

connx = floor((conxhi-conxlo-2*shift)/contourRes)+1;
conny = floor((conyhi-conylo)/contourRes)+1;
connz = floor((conzhi-conzlo)/contourRes)+1;

conx = repmat((conxlo+shift:contourRes:conxhi-shift)',conny,1);
cony = repmat((conylo:contourRes:conyhi),connx,1);
cony = cony(:);

conx = repmat(conx,connz,1);
cony = repmat(cony,connz,1);
conz = repmat((conzlo:contourRes:conzhi),connx*conny,1);
conz = conz(:);

contourPoints = zeros(size(conx,1),3);
count = 0;
for i = 1:size(conx)
    % Only include points in the desired area
    if ((conx(i)-shift*(cony(i)-conylo)/(conyhi-conylo) >= conxlo) && ...
            conx(i) + shift*(conyhi-cony(i))/(conyhi-conylo) <= conxhi)
        count = count + 1;
        contourPoints(count,:) = [conx(i),cony(i),conz(i)];
    end
end
contourPoints = contourPoints(1:count,:);

% Plotting (show the contour )
plot3(conx,cony,conz,'k.');
parcor = [conxlo,conylo,conzlo; conxlo+shift,conyhi,conzlo; conxhi,conyhi,conzlo; conxhi-shift,conylo,conzlo; conxlo,conylo,conzlo];
hold on;
plot3(parcor(:,1), parcor(:,2), parcor(:,3), 'b-');
hold on;
plot3(contourPoints(:,1), contourPoints(:,2), contourPoints(:,3), 'ro');
hold on;
plot3(x(:,1),y(:,1),z(:,1),'b.');

disp('Calculating contour voltages');
contourPotential = zeros(count,n_bins,timesteps);
relContourPotential = zeros(count,n_bins,timesteps);
contourVoltage = zeros(count,timesteps);
relContourVoltage2 = zeros(count,timesteps);
contourElecField = zeros(count, timesteps,3);
%%
% for i = 1:count
%     for t = 1:timesteps
%         for j = 1:n_bins
%             contourPos = [contourPoints(i,1)-centx(j,t), contourPoints(i,2)-centy(j,t), contourPoints(i,3)-centz(j,t)];
%             contourDist = contourPos/sqrt(sum(contourPos.^2))^3;
%             contourPotential(i,j,t) = sqrt(sum((bin_dipole(j,:,t).*contourDist).^2));
%             relContourPotential(i,j,t) = sqrt(sum((relBinPol(j,:,t).*contourDist).^2));
%             % Electric Field
%             fieldTensor = 3*(contourPos * contourPos') ./ sqrt(sum(contourPos.^2))^5;
%             fieldTensor = fieldTensor - 1/sqrt(sum(contourPos.^2))^3;
%             contourElecField(i,t,:) = contourElecField(i,t,:) + reshape(bin_dipole(j,:,t) * fieldTensor,[1,1,3]);
%         end
%         contourVoltage(i,t) = nansum(contourPotential(i,:,t));
%         relContourVoltage2(i,t) = nansum(relContourPotential(i,:,t));
%     end
% end
contourPointsr = reshape(contourPoints',1,3,[]);
contourPosminus = contourPointsr - [centx(:) centy(:) centz(:)];
den2 = sqrt(sum(contourPosminus.^2,2));
contourDistr = contourPosminus./den2.^3;

bin_dipoler = reshape(permute(bin_dipole,[1 3 2]),[],3,1);
contour_cal = sqrt(sum((bin_dipoler.*contourDistr).^2,2));

relBinPolr = reshape(permute(relBinPol,[1 3 2]),[],3,1);
relContour_cal = sqrt(sum((relBinPolr.*contourDistr).^2,2));

nfieldTensor = 3*den2.^(-3) - 1/(den2.^3);

for i = 1:count
    for t = 1:timesteps
        for j = 1:n_bins
            loop_index = n_bins*(t-1)+j;
            contourPotential(i,j,t) = contour_cal(loop_index,:,i);
            relContourPotential(i,j,t) = relContour_cal(loop_index,:,i);
            % Electric Field
            fieldTensor = nfieldTensor(loop_index,:,i);
            contourElecField(i,t,:) = contourElecField(i,t,:) + reshape(bin_dipole(j,:,t) * fieldTensor,[1,1,3]);
        end
        contourVoltage(i,t) = nansum(contourPotential(i,:,t));
        relContourVoltage2(i,t) = nansum(relContourPotential(i,:,t));
    end
end
contourVoltage = contourVoltage./(4*pi*8.85*10^(-12)).*(1.60217662*10^(-19)).*(10^10);      % Answer is in Volts.
contourElecField = contourElecField./(4*pi*8.85*10^(-12)).*(1.60217662*10^(-19)).*(10^10);  % Answer is in V/m
relContourVoltage2 = relContourVoltage2./(4*pi*8.85*10^(-12)).*(1.60217662*10^(-19)).*(10^10);    % Answer is in Volts.
relContourVoltage = contourVoltage(:,:) - contourVoltage(:,5);
contourVoltageDiff = relContourVoltage2 - relContourVoltage;

% Seperate the probe atoms from the target atoms. This will help with
% scaling.
disp('Printing Output');
output_data = cat(3,atom_id,type,x,y,z,vx,vy,vz,q,bins,tempbins,mass,...
    velocity,kintemp,bintemp,distx,disty,distz,pol_x,pol_y,pol_z,...
    xbinpol,ybinpol,zbinpol,bin_pol,rel_pol,voltage,rel_voltage, ...
    elecField(:,:,1), elecField(:,:,2), elecField(:,:,3));
% Remove Potential for when it's not needed (debugging)
% output_data = cat(3,atom_id,type,x,y,z,vx,vy,vz,q,bins,tempbins,mass,...
%     velocity,kintemp,bintemp,distx,disty,distz,pol_x,pol_y,pol_z,...
%     xbinpol,ybinpol,zbinpol,bin_pol,rel_pol);

output_data = permute(output_data, [1,3,2]);  % Rearrange columns
probe_data = output_data(bins(:,1)==n_bins,:,:);
target_data = output_data(bins(:,1)~=n_bins,:,:);

for t=1:timesteps
    % Write target data to file.
    filename = strcat(output_dir, output_file_prefix, 'Results-', ...
        sprintf('%04d',t-1), '.csv');
    output = array2table(target_data(:,:,t));
    columnNames = {'atomID', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz', ...
        'q', 'bin', 'tempbin', 'mass', 'velocity', 'kintemp', 'bintemp', ...
        'distx', 'disty', 'distz', 'xpol', 'ypol', 'zpol', 'xbinpol', ...
        'ybinpol', 'zbinpol', 'binpol', 'rel_pol', 'voltage', ...
        'rel_pot', 'ElecFieldX', 'ElecFieldY', 'ElecFieldZ'};
% Remove Potential for when it's not needed (debugging)    
%     columnNames = {'atomID', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz', ...
%         'q', 'bin', 'tempbin', 'mass', 'velocity', 'kintemp', 'bintemp', ...
%         'distx', 'disty', 'distz', 'xpol', 'ypol', 'zpol', 'xbinpol', ...
%         'ybinpol', 'zbinpol', 'binpol', 'rel_pol'};
    output.Properties.VariableNames = columnNames;
    writetable(output,filename);
    
    % Write probe data to a separate file.
    probe_filename = strcat(output_dir, output_file_prefix, ...
        'Probe-', sprintf('%04d',t-1), '.csv');
    probe_output = array2table(probe_data(:,:,t));
    probe_output.Properties.VariableNames = columnNames;
    writetable(probe_output,probe_filename);
    
    % Write contour data to another file
    contour_filename = strcat(output_dir, output_file_prefix, ...
        'Contour-', sprintf('%04d',t-1), '.csv');
    contour_output = array2table([contourPoints(:,:), contourVoltage(:,t), ...
        relContourVoltage(:,t), relContourVoltage2(:,t), contourVoltageDiff(:,t), ...
        contourElecField(:,t,1), contourElecField(:,t,2), contourElecField(:,t,3)]);
    contour_output.Properties.VariableNames = {'x', 'y', 'z', 'potential', ...
        'rel_pot', 'rel_pot2', 'rel_pot_diff', 'ElecFieldX', 'ElecFieldY', 'ElecFieldZ'};
    writetable(contour_output,contour_filename);
end

disp('Done!');

toc