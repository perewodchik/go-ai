import torch
import concurrent.futures
from ai.network import GoNetwork
from config import Config

def worker(kwargs):
    net = kwargs['net']
    t = torch.randn(1, 9, 9, 9)
    return net(t)[0].sum().item()

if __name__ == '__main__':
    net = GoNetwork(9, 9, 4, 64, 64)
    net.eval()
    tasks = [{'net': net} for _ in range(4)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(worker, t): i for i, t in enumerate(tasks)}
        for fut in concurrent.futures.as_completed(futs):
            print(futs[fut], fut.result())
