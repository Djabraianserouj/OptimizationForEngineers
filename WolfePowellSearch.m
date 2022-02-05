function [t] = WolfePowellSearch(f, x, d, sigma, rho, verbose)
  %WOLFEPOWELLSEARCH Find stepsize t get sufficient decrease and steepness for multidimensional objective along line
  
  %% Purpose:
  % Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x)'*d and
  % gradf(x+t*d)'*d >= rho*gradf(x)'*d
  
  %% Input Definition:
  % f: function handle of type [value, gradient] = f(x).
  % x: column vector in R^n (domain point) 
  % d: column vector in R^n (search direction)
  % sigma: value in (0,1/2), marks quality of decrease. Default value: 1.0e-3
  % rho: value in (sigma,1), marks quality of steepness. Default value:1.0e-2
  % verbose: bool, if set to true, verbose information is displayed
  
  %% Output Definition:
  % t: t is set, such that t satisfies both Wolfe-Powell conditions
  
  %% Required files:
  % <none>
  
  %% Test cases:
  % [t] = WolfePowellSearch(@(x)simpleValleyObjective(x,[0;1]), [-1.01;1], [1;1], 1.0e-3, 1.0e-2, true);
  % should return
  % t=1;
  %
  % [t] = WolfePowellSearch(@(x)simpleValleyObjective(x,[0;1]), [-1.2;1], [0.1;1], 1.0e-3, 1.0e-2, true);
  % should return
  % t=16;
  %
  % [t] = WolfePowellSearch(@(x)simpleValleyObjective(x,[0;1]), [-0.2;1], [1;1], 1.0e-3, 1.0e-2, true);
  % should return
  % t=0.25;
  %
  % [t] = WolfePowellSearch(@(x)nonlinearObjective(x), [0.53;-0.29], [-3.88;1.43], 1.0e-3, 1.0e-2, true);
  % should return
  % t=0.0938;
  
  %% Input verification:
 
  try
    [value, gradient] = f(x);
  catch
    error('evaluation of function handle failed!'); 
  end
  
  if (gradient'*d>= 0)
    error('descent direction check failed!');    
  end 
  if (sigma <= 0 || sigma >= 0.5)
    error('range of sigma is wrong!');    
  end  
  if (rho <= sigma || rho >= 1)
    error('range of rho is wrong!');    
  end   
  if nargin < 6
    verbose = false;
  end
  
  %% Implementation:
  % Hints: 
  % 1. Whenever t changes, you need to update the objective value and
  % gradient properly!
  % 2. Use the return keyword (see documentation)
  if verbose
    disp('Start WolfePowellSearch...');
  end
      
%Complete the code

  t=1;
  [value, gradient] = f(x);
  [value2, gradient2] = f(x+t*d);
  
  if ( value2 > value + t*sigma*gradient'*d)      %Step 6
    disp ('backtracking');  
    t = t/2;
    [value2, gradient2] = f(x+t*d);
    while (value2 > value + t*sigma*gradient'*d)
      disp ('backtracking inner-loop');
      t = t/2; 
      [value, gradient] = f(x);
      [value2, gradient2] = f(x+t*d);
    endwhile
    t_minus = t;
    t_plus = 2*t;
    
  elseif (gradient2'*d >= rho*gradient'*d)   #Step 7
    t
    return 
  
  else        #Step 8
    t=2*t;
    [value2, gradient2] = f(x+t*d);
    while (value2 <= value + t*sigma*gradient'*d)
      disp ('fronttracking');
      t = 2*t;
      [value, gradient] = f(x);
      [value2, gradient2] = f(x+t*d);
    endwhile
      t_minus = t/2;
      t_plus = t;
  endif
  
  t = t_minus;        #Step 9
  [value2, gradient2] = f(x+t*d);
#NOTE For Dr. Hild: I had some problems with the anonymous function for W2, where the values of the gradient2
#weren't updated correctly. So I had to recall fun2 for the rifining case. I hope that I used the anonymous functions correctly.
  
  [value, gradient] = f(x);
  [value2, gradient2] = f(x+t*d);
  while (gradient2'*d < rho*gradient'*d)       #Step 10 
    disp ('Refining');
    t = (t_minus+t_plus)/2;
    [value, gradient] = f(x);
    [value2, gradient2] = f(x+t*d);
    if (value2 <= value + t*sigma*gradient'*d) 
      t_minus = t;
    else t_plus = t;
    endif
  endwhile
  t = t_minus
  if verbose
    disp(sprintf('WolfePowellSearch terminated with t=%d',t));
  end
  
end

