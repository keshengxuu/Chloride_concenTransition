% Figure 1 is a 2D mean field model, showing the qualitative simularity 
% between the model and real data (You need to zoom in to show a subset of 
% the recordings so that the two looks alike)

% Lay out the field
O = MeanFieldModel('Exp1Template');

% Build the round mask (round boundary condition)
[xx,yy] = meshgrid( -(O.n(2)/2)+0.5:(O.n(2)/2)-0.5, -(O.n(1)/2)+0.5:(O.n(1)/2)-0.5);
xx = xx/O.n(2); % Normalized spatial unit
yy = yy/O.n(1); % Normalized spatial unit
rr = sqrt(xx.^2 + yy.^2); % Normalized spatial unit
mask = rr < 0.5; % The easiest way to apply mask is to redefine its activation function    
f_original = O.param.f;
O.param.f = @(v) mask.*f_original(v); % Now neurons outside the boundary can not fire

%% Build recurrent projections
[ P_E, P_I1, P_I2 ] = StandardRecurrentConnection( O );

%% External input
Ic = 200;
stim_t = [2 5]; % second
stim_x = [0.5 0.1]; % normalized spatial unit
stim_r = 0.05; % normalized spatial unit
O.Ext = ExternalInput;
O.Ext.Target = O;
O.Ext.Deterministic = @(x,t) ( (sqrt(((x(:,1)-1)/(O.n(2)-1)-stim_x(1)).^2 + ((x(:,2)-1)/(O.n(1)-1)-stim_x(2)).^2)) < stim_r) .* ...
                             ( (stim_t(2)*1000) > t & t > (stim_t(1)*1000)) .* ...
                             Ic; % x: position (neuron index), t: ms, unit: pA 

%% Simulation settings 
dt = 1; % ms
write_cycle = 10; % Every 10 ms save the result once to avoid asking for too much RAM 
R = CreateRecorder(O,10000); % The 2nd argument is Recorder.Capacity 
T_end = write_cycle*R.Capacity - 1*write_cycle; % simulation end time.                           

%% Realtime plot setting
flag_realtime_plot = 1; % whether you want to see simulation result real time or not
T_plot_cycle = 1000; % How often updating the figures

%% Simulation
while 1 % You can press 'q' to escape this while loop     
    % End mechanism
    if O.t >= T_end
        break;
    end

    if O.t ==1
        figure; 
        h1 = imagesc(O.R.f); 
        colorbar;
        % title('V');
        clim([0 0.12]); % 设置第一个图的颜色条范围为 -80 到 -20
        axis equal; % 设置子图的横纵比为1:1
        xlim([1 100]); % 设置横坐标范围从 0 到 100
        ylim([1 100]); % 设置纵坐标范围从 0 到 100
        % ---- 自定义 colormap（仿 PRGn 配色）----
        cmap = [ ...
            64, 0, 75;        % 深紫
            118, 42, 131;     % 紫色
            153, 112, 171;    % 淡紫
            194, 165, 207;    % 浅紫
            231, 212, 232;    % 过渡浅灰
            247, 247, 247;    % 中性色
            217, 240, 211;    % 淡绿
            166, 219, 160;    % 浅绿
            90, 174, 97;      % 绿色
            27, 120, 55];     % 深绿

        cmap = cmap ./ 255; % 归一化到 [0,1]
        colormap(interp1(linspace(0,1,size(cmap,1)), cmap, linspace(0,1,256)));
    end

    Update(O,dt);

    if mod(O.t,T_plot_cycle) < dt && flag_realtime_plot 
        % f=plot(O);drawnow
        set(h1, 'CData', O.R.f);drawnow
    
    end 
end
