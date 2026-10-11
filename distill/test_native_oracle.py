import unittest

import torch
from infinite_tensor import InfiniteTensor,TensorWindow

from distill.native_oracle import mix_channels,mixed_tensor


class NativeOracleTests(unittest.TestCase):
    def test_swap_normalizes_each_sources_own_weight(self):
        student=torch.cat((torch.arange(5.)[:,None,None].expand(5,64,64)*2,
                           torch.full((1,64,64),2.)))
        teacher=torch.cat(((10+torch.arange(5.))[:,None,None].expand(5,64,64)*3,
                           torch.full((1,64,64),3.)))
        a=mix_channels(student,teacher,'teacher_height')
        torch.testing.assert_close(a[:4],student[:4]/2)
        torch.testing.assert_close(a[4],teacher[4]/3)
        b=mix_channels(student,teacher,'teacher_features')
        torch.testing.assert_close(b[:4],teacher[:4]/3)
        torch.testing.assert_close(b[4],student[4]/2)

    def test_nonoverlapping_oracle_storage_retains_global_coordinates(self):
        def source(scale):
            window=TensorWindow(size=(6,64,64),stride=(6,32,32))
            def forward(contexts):
                rows=[]
                for _,y,x in contexts:
                    yy=torch.arange(y*32,y*32+64)[:,None]
                    xx=torch.arange(x*32,x*32+64)[None,:]
                    field=(yy+xx).float()*scale
                    rows.append(torch.cat((field[None].expand(5,64,64)*2,torch.full((1,64,64),2.))))
                return rows
            return InfiniteTensor(shape=(6,None,None),f=forward,output_window=window,batch_size=16)
        a,b=source(1.),source(2.)
        result=mixed_tensor(a,b,'teacher_height',None)[:, -64:64,-64:64]
        target=b[:, -64:64,-64:64]
        candidate=a[:, -64:64,-64:64]
        torch.testing.assert_close(result[4],target[4]/target[-1],rtol=0,atol=0)
        torch.testing.assert_close(result[:4],candidate[:4]/candidate[-1:],rtol=0,atol=0)
        torch.testing.assert_close(result[-1],torch.ones(128,128),rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
