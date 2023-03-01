# PINN
This repository contains the implementations of PINNs for a few standard equations and related work/discussion.

<br/>

## 1 Implementation of PINNs

<br/>

### A 

In this section of the equation, we have three files.

Burgers.ipynb
Burgers_Inference.ipynb
Burgers_Discovery.ipynb

For burgers equation the analytical solution (y values) are picked up from burgers-shock.mat, used this data file in the original repository for Burgers_Inference.ipynb and Burgers_Discovery.ipynb

The inference and discovery notebooks import the exact solution (analytic solution) from a the data, using which we can calculate the relative L2 error between the exact solution and the one predicted by the network.

In the data-driven discovery method, we have the values for the lambda1 and the lambda2 set by the user initially which the model learns and in turns predicts the equation. We can choose them randomly also, like in the burgers equation, we have them set to 0 and -6, whereas the actual values are 1 and -0.00318 (Viscosity)
The model learns the lambda values just as it learns other parameters. The closer the values are to the actual values in the equation the better the model has understood the equation.

The only difference between the discovery and the inference setups is that we add parameters to the equation in the discovery setup which the model learns and helps formulate the equation.
The data generation and usage and plit between train/test is very similar. We spread a unifrom mesh over the problem sample space.

In Burgers.ipynb, we have the training only based on the initial conditions, boundary condition points, the y values (predictions) based on the input data mesh. We do not have analytic/exact solution y values and can't measure the performance. This is an exploratory notebook to just visualize the predictions. 

X values generated through a simple meshgrid or lhs (latin hypercube sampling) (in case of discovery) and used alongwith those read from the file.

The entire huge set is used for the testing, whereas for the training, we use a small randomly sampled subset of the same generated set.





<br/>

### B

Didn’t have the data file for diffusion or kovasznay flow, so generated the data by the exact (analytical) solution.
The sizes of the data are chosen such that the execution completes and we don't get memory exceeded.
For more than the current;;y used sizes of points got a Memory error on Colab (free GPU version) as well as locally. Could have setup an appropriate batch size but didn't explore that option.

All other considerations and settings are similar to those for the burgers equations.

<br/>

#### Diffusion Equation

Here we have sampled 500 points for training randomly from the test set

For the inference, we have 200x100 points for inference. This can be decided by using appropriate values of h and k.
The diffusion_pinn.mat contains the y values for 200x100 data points.  


<br/>

#### Kovasznay Flow 
Here we have sampled 200 points for training randomly from the test set

The training setup points on the boundary and initial conditions sampled from the 100 points and initial conditions for the Diffusion.ipynb

The kovasznay_pinn.mat contains the y values for 200x200 data points. This can be decided by using appropriate values of h and k. 



The analytical solution for v differs from the one mentioned on [DeepXDE](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html) in the 2 papers, [PINN for NS](https://arxiv.org/pdf/2104.06217.pdf) and [Numerical Approximation of NS](https://pdf.sciencedirectassets.com/272570/1-s2.0-S0021999119X00108/1-s2.0-S0021999119301950/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEN7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQCIqeDCWoixMw6zTFLNtVKI%2BoWYp89M8au8vGtBW6DPxwIgObTdtrpqddd%2FFr94%2F8zc%2FrkdFix9eB59XLnFqOkeSbEqvAUIh%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDAvYnEeXZhv8s1%2BW8SqQBQOrvXdrYHrI6ee2XmkoUcEwU8JyuVCy%2B82R5oNluK411Ncber8%2BJh9INrvvuedxl0vqSTSxAzPsofg8YTlnMCcYK6IVSwizQf%2BNzvKBsINtOMBkIY3GwS03W72qRm2MIqTaygwl927u%2BZB8Ro2eB06UjOQ0%2F6bmQuzaqNxyPzO8xRZuPRD5jk4fdC4yz%2B9CZ%2FhYOhod1GSun1M%2BIX46lI6av5xrqmF1R1sGYCsORI366Ep04SK5i6iDXiqUmnMskhUTGgdONC8ly1StqTQMeG4ZmXx6pSIT8A8Kn1XDc1D3Z7il3KEX8BZ%2FfEaBMKLJTkEMZ5Y3LMzpTWYROIb9oiHv7lRAl5PjElRWFunAb%2BzVTj%2Bz7aiAk%2FY14ioEleNw91X7Tqp1m21kpKjC34SOW47hSnal4Lxc8hLsPAKNA8xBOgAa4mvHwn2TtzQm2IS4PZjeqBgG6KLJyD4U7MJCt8ECYkKJhVXDBosMr1v%2BiOO74t0NOUbeUCvqzBR%2BWR96rZ6ro4uT5cJKU3PbaLH4vRZSZyV7yS2Tn%2F%2Fl4%2B1TFeHS5DLslOz%2Bx3596pX%2F6y2Wzuhkc2ntXedzB2PFOTBkiDAaHXgxELKSojXBbaBo%2Beni%2FiYPEQzdie9xqrL5rscf0rtUgbrcIdBpukwsam%2BjlenXQsv8d9xcworbKIPZA%2BSAsEC%2BHYXOUxN6hTvvtSaMit2QHGm1Gnk%2BZj1odhYXr1HVWQEeUFcU6bXPeJC43DXJwHzeXqTB%2Fs%2Fz0horufnxWcahHU5riGkXOKGhAvv3PQzFiq0q%2FVlLUJTmWknszhRppbcBojW%2FlNTP4NZ56kDgVOTQh37ThSl1Z7%2BVYBH%2FG0qG79IEV0b0B%2Fef6cuGBI9lMLXC%2B58GOrEB6FcoZXvRjNaujOz5Pq0Pzk2nRgATGmZkDf4Gp4oX8DN8KO2OQfOum2MYySwX0TwE6iyEneNM1MbVxaHjBhFgB3rqu2OXevPk%2B%2BkFu%2Buomrk0x8XnixsopAFi5upoRIRcd3mNklxYz2NaEJwUlqoesYWVYOlRtGAVS7JF0UuawD76z7RigIUup3aKl02CgojY0Dhx5E7YFfNMY8snYxVqtdpnF1F7F4%2BxnPRpHuZd53QL&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230301T055638Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY2DR7ZPR4%2F20230301%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8793b9b6158eab373f96fe88d1fa7d5fc8c9df38e779da9958d22a4a82dca706&hash=f9e2c35dd6a656083350f4f91192671b4c07922d61a034c75a04540d01b9410f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0021999119301950&tid=pdf-25ae7bff-6e3d-46e2-a1c1-e49735c57c04&sid=9ebfe6ec3a932246661a2422565dd48b620cgxrqb&type=client), used the one mentioned in the papers.

<br/>

## 2 Paper Summary/Discussion

<br/>

Linear ODEs only, as mentioned in the paper, expanding to spatial or spatio-temporal PDEs with Dirichlet boundary conditions would be challenging.

Linked ODE Residulas to absolute error showed that we could bound absolute error by a function of residuals.

