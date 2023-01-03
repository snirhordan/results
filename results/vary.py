# This version supports varying-size-sets as inputs

import torch
import torch.nn as nn
import numpy as np
import scipy as sp
import torch.nn.functional as F


def tr12(input):
   return torch.transpose(input, 1, 2)


class embed_vec_sort(nn.Module):
   # Calculates a permutation-invariant embedding of a set of vectors in R^d, using Nadav
   # and Ravina's sort embedding. It has two modes of operation:
   #
   # Constant set size: (varying_set_size = False)
   # Input size: b x d x n, with b = batch size, d = input feature dimension, n = set size
   # Output size: b x d_out. Default d_out is 2*d*n+1
   #
   # Varying set size: (varying_set_size = True)
   # Here the input can contain any number of vectors. 
   # n-1 is taken to be the number of linear regions in the weight function, and should be
   # larger or equal to the maximal input set size. 
   # Input size: b x (d+1) x n, with b = batch size, d = input feature dimension
   # Output size: b x d_out. Default d_out is 2*d*n+1
   # The first row of the input should contain 1 or 0, corresponding to existing or absent input vectors.
   #
   # The weight interpolation was taken as in the paper:
   # Learning Set Representations with Featurewise Sort Pooling, by Zhang et al., 2020   
   #
   # learnable_weights - Treats the weights as learnable parameters
   def __init__(self, d, n, d_out = None, varying_set_size = False, learnable_weights = False):
      super().__init__()
      
      if d_out is None:
         d_out = 2*d*n+1

      self.d = d
      self.n = n
      self.d_out = d_out
      self.varying_set_size = varying_set_size

      if learnable_weights:
         self.A = nn.Parameter(torch.randn([d, d_out], requires_grad=True))
         self.w = nn.Parameter(torch.randn([1, n, d_out], requires_grad=True))
      else:
         self.A = torch.randn([d, d_out], requires_grad=False)
         self.w = torch.randn([1, n, d_out], requires_grad=False)
         self.register_buffer(name="sort-embedding matrix A", tensor=self.A)
         self.register_buffer(name="sort-embedding vector w", tensor=self.w)

   # Input: A vector k of size b, containing the set size at each batch
   # Output: A weight tensor w_out of size b x n x d_out, which is an interpolated
   #         version of w, such that w_out[r, :, t] corresponds to the weight vector self.w[0, :, t]
   #         after interpolation to the set size k[r].
   def interpolate_weights(self, k):
      b = torch.numel(k)

      # i corresponds to the sorted sample number in the output weight w_out
      i = (torch.arange(1, 1+self.n)).unsqueeze(0).unsqueeze(2)

      # j corresponds to the sorted sample number in the input weight self.w
      j = (torch.arange(1, 1+self.n)).unsqueeze(0).unsqueeze(0)

      # k contains the set size of each batch
      # Note that the roles of k and n here are replaced compared to those in the paper.
      k = k.unsqueeze(1).unsqueeze(1)
      
      interp_matrix = torch.clamp(1-torch.abs( (i-1)*(self.n-1)/(k-1) - (j-1) ), min=0)     
      w_out = torch.bmm(interp_matrix, self.w.repeat(b,1,1))

      return w_out

   def forward(self, input):
      if self.varying_set_size:
         assert input.shape[1] == self.d+1, 'Dimension 1 of input should be d+1'
         q = input[:,0,:]
         q = q.unsqueeze(1)
         X = input[:,1:,:]
      else:
         assert input.shape[1] == self.d, 'Dimension 1 of input should be d'
         X = input

      assert input.shape[2] == self.n, 'Dimension 2 of input should be n'

      prod = tr12( torch.tensordot( tr12(X), self.A, [[2], [0]] ) ) 
      prod[q.repeat(1,self.d_out,1) == 0] = np.inf

      if self.varying_set_size:
         k = torch.sum(q, dim=2).flatten()
         w = self.interpolate_weights(k)
      else:
         w = self.w   
      
      [prod_sort, inds_sort] = torch.sort(prod, dim=2)
      prod_sort[prod_sort == np.inf] = 0

      out = torch.sum( prod_sort * tr12(w), dim=2)

      return out


# Calculates an embedding of n vectors in R^d that is invariant to permutations,
# rotations and optionally translations.
#
# Input size: b x (d + d_feature) x n
# b: batch size
# d: dimension of Euclidean space (currently only d=3 is supported)
# d_feature: accompanying feature dimension (default: 0)
# n: number of points
#
# Input shape: input[:,  0:(d-1), :] should be the Euclidean coordinates. The rest input[:,  d:(d+d_feature), :] should contain the accompanying feature vectors.
#
# translation_invariant: When set to true, applies mean-centering to the data points, leading to translation invariance.
# is_compact: When set to true, the embedding dimension is 2*(d+d_feature)*n + 1.
#             Otherwise the dimension is much larger: 2 * ( d*d + (d-1)*d_feature + 2*(d+d_feature)*(n-(d-1)) + 1 ) * n! / (n-d+1)!
class embed_graph(nn.Module):
   def __init__(self, d, n, d_feature = 0, translation_invariant = True, is_compact = True):
      super().__init__()

      self.d = d      
      self.n = n
      self.d_feature = d_feature

      self.translation_invariant = translation_invariant

      self.n_combs = np.prod(range(n-d+2,n+1))

      # Dimension of the embedding vector for one index-combination
      self.d_comb = d*d + (d-1)*d_feature + 2*(d+d_feature)*(n-(d-1)) + 1

      # Dimension of the embedding for the entire graph
      if is_compact:
         self.d_graph = 2*(d+d_feature)*n + 1
      else:
         self.d_graph = 2*self.d_comb*self.n_combs
      
      self.embed_comb = embed_vec_sort(d+d_feature, n-(d-1))
      self.embed_graph = embed_vec_sort(self.d_comb, self.n_combs, self.d_graph)


   def forward(self, input):
      [b,d_tot,n] = input.shape
      d_feature = self.d_feature
      d = d_tot - d_feature
      
      assert(d == self.d and n == self.n)
      assert(d == 3) # Only d=3 is currently supported

      X = input[:,range(d),:]
      F = input[:,range(d,d_tot),:] if d_feature > 0 else None

      if self.translation_invariant:
         centers = torch.mean(X, axis=2)
         centers = torch.unsqueeze(centers, 2)
         X -= centers

      combs = torch.combinations(torch.arange(start=0, end=n), r=d-1, with_replacement=False)
      combs_flip = combs.flip(dims=[1])
      combs = torch.cat([combs, combs_flip], dim=0)

      global_vecs = torch.zeros([b, self.d_comb, self.n_combs])

      for i, comb in enumerate(combs):
         M0 = X[:, :, comb]
         xprod = torch.linalg.cross(M0[:,:,0], M0[:,:,1], dim=1)
         xprod = torch.unsqueeze(xprod, 2)

         # For a vector combination (v1, v2), the matrix M consists of columns [v1, v2, v1 x v2]
         M = torch.cat([M0, xprod], dim=2)
         Mt = tr12(M)
         MtM = torch.bmm(Mt,M)
         MtM_vec = torch.flatten(MtM, 1)

         if F is None:
            vec1 = MtM_vec
         else:
            vec1 = torch.cat([MtM_vec, torch.flatten(F[:,:,comb] , 1)], dim=1)
         
         # Complement of the current combination
         combcomp = [x for x in range(self.n) if x not in comb]

         # Product of M^T with all vectors except v1,v2
         MtV = torch.bmm(Mt, X[:,:,combcomp])

         if F is None:
            vecs_to_embed = MtV
         else:
            vecs_to_embed = torch.cat([MtV, F[:,:,combcomp]], dim=1)

         vec2 = self.embed_comb(vecs_to_embed)

         global_vecs[:, :, i] = torch.cat([vec1, vec2], dim=1)

         out = self.embed_graph(global_vecs)

      return out


# Tests the X_p, X_m counterexample
def test_X_pm():
   # Tells if we use the z atom-number feature in addition to the Euclidean coordinates
   use_features = True

   X_p = torch.tensor([[ 6.73555740e-17,  1.10000000e+00,  2.20000000e+00],
      [-2.02066722e-16, -1.10000000e+00, -2.20000000e+00],
      [ 2.03951216e+00,  6.27697301e+00,  0.00000000e+00],
      [-2.03951216e+00, -6.27697301e+00,  0.00000000e+00],
      [-3.30000000e+00,  4.04133444e-16,  4.40000000e+00],
      [ 3.30000000e+00, -8.08266887e-16, -4.40000000e+00]])

   # Atom number
   F_p = torch.tensor([[[8.0,1.0,1.0,8.0,1.0,1.0]]])
   F_m = F_p

   X_m = torch.tensor([[ 6.73555740e-17,  1.10000000e+00, -2.20000000e+00],
      [-2.02066722e-16, -1.10000000e+00,  2.20000000e+00],
      [ 2.03951216e+00,  6.27697301e+00,  0.00000000e+00],
      [-2.03951216e+00, -6.27697301e+00,  0.00000000e+00],
      [-3.30000000e+00,  4.04133444e-16,  4.40000000e+00],
      [ 3.30000000e+00, -8.08266887e-16, -4.40000000e+00]])

   X_p = tr12(torch.unsqueeze(X_p, 0))
   X_m = tr12(torch.unsqueeze(X_m, 0))

   b = 1 # Number of batches
   d = 3 # Vector dimension. Only works for d=3.
   n = X_p.shape[2] # Number of points in each batch

   d_feature = 1 if use_features else 0
            
   embed_vecs = embed_vec_sort(d,n)
   embed_g = embed_graph(d, n, translation_invariant=True, d_feature=d_feature, is_compact = True)

   if use_features:
      P = torch.cat([X_p, F_p], dim=1)
      Q = torch.cat([X_m, F_m], dim=1)
   else:
      P = X_p
      Q = X_m

   # Invariance check
   #perm = torch.randperm(n) 
   #Q = P[:,:,perm]
   #Q[0,0,0] += 0.01 # Screws the x coordinate of the first vector a little bit
   #Q[0,3,0] *= 1.1 # Screws the feature of the first vector a little bit

   P_embed = embed_g(P)
   Q_embed = embed_g(Q)

   diff = torch.norm(P_embed-Q_embed, dim=1).numpy() / torch.norm(P_embed, dim=1).numpy()

   print('\nRelative difference per batch:', diff)
   print('')



# Synthetic example with multiple batches
def test_synthetic():
   b = 4 # Number of batches
   d = 3 # Vector dimension. Only works for d=3.
   n = 25 # Number of points in each batch

   embed_g = embed_graph(d, n, translation_invariant=True)

   P = torch.randn([b, d, n])
   R = torch.zeros([b, d, d])

   # Create random rotation matrix R
   for r in range(b):
      A = np.random.normal(size=[3,3])
      U, S, Vh = np.linalg.svd(A, full_matrices=True)
      R[r,:,:] = torch.from_numpy(U)
      #input2[r,:,:] = torch.prod(R, input[r,:,:])

   Q = torch.bmm(R, P)

   # Mess with the first vector of the first batch of Q
   Q[0,:,0] = Q[0,:,1]

   # Apply a random global translation on the second batch of Q
   Q[1,:,:] = Q[1,:,:] + 10.0*torch.randn([1, d, 1])

   # Apply a random permutation on the second and third batch of Q
   perm = torch.randperm(n) 
   Q[1,:,:] = P[1,:,perm]
   Q[2,:,:] = P[2,:,perm]

   P_embed = embed_g(P)
   Q_embed = embed_g(Q)

   diff = torch.norm(P_embed-Q_embed, dim=1).numpy() / torch.norm(P_embed, dim=1).numpy()

   print('\nRelative difference per batch:', diff)
   print('')



# Synthetic example with multiple batches
def test_varying_size_embedding():
   b = 2 # Number of batches
   d = 30 # Vector dimension. Only works for d=3.
   n = 5 # Number of points in each batch

   embed = embed_vec_sort(d, n, varying_set_size = True, learnable_weights = True)

   X = torch.randn([b, d, n])
   #Y = X

   q = torch.from_numpy(np.array([1])).repeat(b,1,5)
   X = torch.cat([q,X], dim=1)

   k = torch.from_numpy(np.array([2,3,4,5]))
   print(k)
   embed.interpolate_weights(k=k)

   X_embed = embed(X)

   print(X_embed)

   # Apply a random permutation on the second and third batch of Y
   #perm = torch.randperm(n) 
   #X[1,:,:] = P[1,:,perm]
   #Q[2,:,:] = P[2,:,perm]

   #X_embed = embed(X)
   #Q_embed = embed_g(Q)

   #diff = torch.norm(P_embed-Q_embed, dim=1).numpy() / torch.norm(P_embed, dim=1).numpy()

   #print('\nRelative difference per batch:', diff)
   #print('')


if __name__ == '__main__':
   test_varying_size_embedding()
   #test_X_pm()
   #test_synthetic()