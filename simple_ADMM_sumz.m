%% %%%%%%%%%%%%%%%%%%%%%%%%%%%   Proposed Algorithm    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
SliceNum = 5
UENum = 10
ResNum = 1
z = zeros(SliceNum,1);
u = zeros(SliceNum,1);
x = zeros(SliceNum,UENum);
Rmax = 100;
alpha = randi([1,99],SliceNum,UENum)/100;
rho = 10

%tmpz = abs(randn(SliceNum,UENum));
%z = tmpz;


MaxIter = 5;
for ite = 1:MaxIter
    % X-update
    for i=1:SliceNum
        [utility(i,:),x(i,:)] = simple_convex_alogrithm(UENum,u(i),z(i),rho,alpha(i,:),Rmax);
    end
    % Z-update
    %for j = 1: ResNum
        H = 2*eye(SliceNum);
        f = zeros(SliceNum,1);
        sumx = sum(x,2)
        for i=1:SliceNum
            f(i) = -2*( sumx(i) + u(i) );
        end
        Aeq = ones(1,SliceNum);
        beq = Rmax;
        lb = zeros(1,SliceNum);
        ub = Rmax*ones(1,SliceNum);
        z= quadprog(H,f,[],[],Aeq,beq,lb,ub);

    %end
    % U-update
    u = u + ( sum(x,2) - z );

    % history 
    sum_utility(ite) = sum(sum(utility));
    sum_gap(ite)  = ((sum(sum(x)))-Rmax)./Rmax;

    disp(ite)
    z-u
end

figure(10);plot(sum_utility);
figure(11);plot(sum_gap);
disp('proposed algorithm compelted.')


function [utility,X] = simple_convex_alogrithm(UENum,u,z,rho,alpha,Rmax)

xmin = ones(1,UENum);
xmax = Rmax;

cvx_begin

variable x(UENum)

    for i = 1:UENum
        y(i) = x(i)^(alpha(i))/alpha(i);
    end
    fx = sum(y) - 0.5 * rho * norm(  sum(x) - z + u );
    minimize( -fx )
    subject to
        y >= xmin
        sum(x) < xmax
cvx_end


for i = 1:UENum
   utility(i) = x(i)^(alpha(i))/alpha(i);
end

X = x;

end