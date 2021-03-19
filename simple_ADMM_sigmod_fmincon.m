%% %%%%%%%%%%%%%%%%%%%%%%%%%%%   Proposed Algorithm    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
load matlab.mat

SliceNum = 3
UENum = 5
ResNum = 1
z = zeros(SliceNum,1);
u = zeros(SliceNum,1);
x = zeros(SliceNum,UENum);
Rmax = 100;
rho = 5

%% %%%%%%%%%%%%%%%%%%%%%% static allocation #######################################
x_static = Rmax/SliceNum/UENum * ones(SliceNum,UENum);

sumY = total_utility(SliceNum,UENum,alpha,weight,Rmax,x_static);


%% %%%%%%%%%%%%%%%%%%%%%% optimization allocation #######################################

MaxIter = 20;
for ite = 1:MaxIter
    % X-update
    for i=1:SliceNum
        fun = @(x)simple_fmincon_alogrithm_in(x, u(i,:),z(i,:),rho,alpha(i,:),weight(i,:),Rmax,UENum);
        const = @(x)constraint_in(x,alpha(i,:),weight(i,:),Rmax,UENum);
        options = optimset('fmincon');
        options = optimset(options, 'MaxFunEvals', 1e3);
        x(i,:) = fmincon(fun,ones(1,UENum),[],[],[],[],zeros(1,UENum),Rmax*ones(1,UENum),const,options);
        utility(i) = simple_fmincon_alogrithm(x(i,:), u(i,:),z(i,:),rho,alpha(i,:),weight(i,:),Rmax,UENum);
    end
    
    sumx = sum(x,2)
    % Z-update
    %for j = 1: ResNum
        H = 2*eye(SliceNum);
        f = zeros(SliceNum,1);
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
    u = u + ( sumx - z );

    % history 
    sum_utility(ite) = sum(sum(utility));
    sum_gap(ite)  = ((sum(sumx))-Rmax)./Rmax;

    disp(ite)
end

sumY = total_utility(SliceNum,UENum,alpha,weight,Rmax,x);

figure(10);plot(sum_utility);
figure(11);plot(sum_gap);
disp('proposed algorithm compelted.')


function  sumY=total_utility(SliceNum,UENum,alpha,weight,Rmax,x)

for i=1:SliceNum
    for j=1:UENum
        Y(i,j) = weight(i,j)*Rmax/(Rmax*exp(-alpha(i,j)*x(i,j)) + 1);
    end
end
sumY = sum(sum(Y));
end




function fx = simple_fmincon_alogrithm_in(x,u,z,rho,alpha,weight,Rmax,UENum)


for i=1:UENum
    y(i) = weight(i)*Rmax/(Rmax*exp(-alpha(i)*x(i)) + 1);
end

fx = - (  sum(y) - 0.5 * rho * abs( sum(x) - z + u )  );
end

function [c,ceq] = constraint_in(x,alpha,weight,Rmax,UENum)

for i=1:UENum
    c(i) = - Rmax/(Rmax*exp(-alpha(i)*x(i)) + 1) + 2.5;
end
ceq=[];
end