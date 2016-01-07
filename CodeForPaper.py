#!/usr/bin/env python
# encoding:utf-8
"""
Created on Feb 15, 2014

@author: Linus

API调用示例
"""
import pymongo
import networkx as net
import matplotlib.pyplot as plot
from scipy.optimize import curve_fit
import numpy as np
import math
import pickle
import time
import heapq
import random
import sys
import datetime
sys.setrecursionlimit(99999)

def savePicName(db_table_name, repeat):
    nowStr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return db_table_name+'_'+nowStr+'_'+str(repeat)+'t'+'.png'

def storeData(db_table_name, repeat, arr):
    nowStr = datetime.datetime.now().strftime("%Y%m%d")
    file_object = open(db_table_name+'_'+nowStr+'_'+str(repeat)+'t'+'.dat', 'a')
    file_object.write(str(arr)+'\n')
    file_object.close()
    
def func(arr,repeat):
    ttt = [x/repeat for x in arr]
    return ttt

def ac(core):
    degrees = net.degree(core)
    edges = net.edges(core)
    m=len(edges)
    zuoshang=0
    you=0
    zuoxia=0
    for edge in edges:
        node1=edge[0]
        node2=edge[1]
        j=degrees[node1]
        k=degrees[node2]
        zuoshang=zuoshang+j*k
        you=you+0.5*(j+k)
        zuoxia=zuoxia+0.5*(j*j+k*k)
    shang=(1.0*zuoshang/m-1.0*(you/m)*(you/m))
    xia=(1.0*zuoxia/m-1.0*(you/m)*(you/m))
    r=1.0*shang/xia
    print "同配系数 ： %f"%r
    #若r<0，则异配  否则同配
    return r

def differencelist(a, b):
    return list(set(a).difference(set(b))) # a中有而b中没有的

def unionlist(a, b):
    return list(set(a).union(set(b)))

def interlist(a, b):
    '''
    worst case: O(max(len(a), len(b))
    '''
    s = set(b)
    return [item for item in a if item in s]

def refine(arr):
    ans = []
    for i in arr:
        ans.append((i[1]["A"], i[1]["B"]))
        ans.append((i[1]["B"], i[1]["A"]))
    return ans

class TopkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []
 
    def Push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)
    def TopK(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in xrange(len(self.data))])]

def init(db_table_name):
    connection = pymongo.Connection('localhost', 27017)
    db = connection.sina
    print db_table_name
    
    collection = db.Zebra

    g = net.Graph()
    c = collection.find()
    for doc in c.batch_size(30):
        i = 0
        obj = next(c, None)
        while obj :
            if obj: 
                # from-to
                fris = []
                fromnode = obj['from']
                tonode = obj['to']
                xyz = (str(fromnode), str(tonode))
                fris.append(xyz)
                g.add_edges_from(fris)

            obj = next(c, None)
            i = i + 1
    return g

def process(db_table_name, repeat):
    before = time.time()
    
    res_random_predictors=[0,0,0,0,0,0,0,0,0,0]
    res_pa=[0,0,0,0,0,0,0,0,0,0]
    res_cn=[0,0,0,0,0,0,0,0,0,0]
    res_ra=[0,0,0,0,0,0,0,0,0,0]
    res_aa=[0,0,0,0,0,0,0,0,0,0]
    res_car=[0,0,0,0,0,0,0,0,0,0]
    res_lcar=[0,0,0,0,0,0,0,0,0,0]
    res_caa=[0,0,0,0,0,0,0,0,0,0]
    res_cra=[0,0,0,0,0,0,0,0,0,0]
    res_cjc=[0,0,0,0,0,0,0,0,0,0]
    res_cpa=[0,0,0,0,0,0,0,0,0,0]
    res_lcaa=[0,0,0,0,0,0,0,0,0,0]
    res_lcra=[0,0,0,0,0,0,0,0,0,0]
    res_lcjc=[0,0,0,0,0,0,0,0,0,0]
    res_lcpa=[0,0,0,0,0,0,0,0,0,0]
    
    
    baifenbi=[2,4,6,8,10,12,14,16,18,20]
    
    for cou in baifenbi:
        for i in range(repeat):
            core = init(db_table_name)  # 初始化网络
            r = ac(core)  # 同配系数r
            node_list = net.nodes(core)  # 网络中所有节点的列表
            print "node count : %d" % len(node_list) 
            edges = net.edges(core)  # 网络中所有连边的列表
            print "edge count : %d" % len(edges) 
            wanted = cou * len(edges) // 100  # 断边的数量
            print "this loop wanted : %d" % wanted
            
            fal = []  # 保存断边的数组
            edge = random.sample(edges,wanted)
            for i in range(len(edge)):
                fal.append(edge[i])
            core.remove_edges_from(fal)  # 从网络总删除断边
            
            random_predictors = []  # 随机补足删除的断边
            edge = random.sample(edges,wanted)
            for i in range(len(edge)):
                random_predictors.append(edge[i])
            
            pa=[]
            cn=[]
            ra = []
            car = []
            aa = []
            caa = []
            cra = []
            cjc = []
            cpa = []
            linuscar = []
            lcaa = []
            lcra = []
            lcjc = []
            lcpa = []
            
            
            for i in xrange(len(node_list)):  # 遍历所有节点列表
                node1 = node_list[i]  # seed node A
                for j in xrange(i, len(node_list)):
                    node2 = node_list[j]  # seed node B
                    if i == j:
                        continue
                    if core.has_edge(node1, node2):
                        continue
                    # PA
                    k1 = core.degree(node1)
                    k2 = core.degree(node2)
                    PA_value = k1 * k2
                    pa.append((PA_value,{"A":node1, "B":node2}))
                        
                    neighlist1 = core.neighbors(node1)  # seed node A 邻居节点列表
                    neighlist2 = core.neighbors(node2)  # seed node B 邻居节点列表
                    cnlist = interlist(neighlist1, neighlist2)  # 交集
                    unionedlist = unionlist(neighlist1, neighlist2) # 并集
                    unioned = len(unionedlist)
                    
                    ex_list = differencelist(neighlist1, cnlist)
                    ex_value = len(ex_list)
                    ey_list = differencelist(neighlist2, cnlist)
                    ey_value = len(ey_list)
                    
                    
                    CN_value = len(cnlist)  # A 和 B 之间邻居节点数量
                    if CN_value == 0:
                        continue
                    else:
                        # CN
                        cn.append((CN_value,{"A":node1, "B":node2}))
                        
                        # AA
                        AA_value = 0
                        for node in cnlist:
                            nodedegree = core.degree(node)
                            AA_value = AA_value + 1.0 / math.log(nodedegree, 2)
                        aa.append((AA_value,{"A":node1, "B":node2}))
                        
                        # RA
                        RA_value = 0
                        for node in cnlist:
                            nodedegree = core.degree(node)
                            RA_value = RA_value + 1.0 / nodedegree
                        ra.append((RA_value,{"A":node1, "B":node2}))
                        
                        small = net.Graph()  # 局部社团里的小网络
                        for i in cnlist:
                            small.add_node(i)
                        
                        # CAR
                        lcl = 0
                        for m in xrange(len(cnlist)):
                            node11 = cnlist[m]
                            for n in xrange(m, len(cnlist)):
                                node22 = cnlist[n]
                                if m == n:
                                    continue
                                if core.has_edge(node11, node22):
                                    lcl = lcl + 1
                                    small.add_edge(node11, node22)
                                else:
                                    continue
                        if lcl == 0:
                            continue
                        
                        CAR_value = CN_value*lcl
                        car.append((CAR_value,{"A":node1, "B":node2}))
                        
                        # CJC 
                        CJC_value = 1.0 * CAR_value / unioned
                        cjc.append((CJC_value,{"A":node1, "B":node2}))
                        
                        # CAA
                        CAA_value = 0
                        for node in cnlist:
                            nodedegree = core.degree(node)
                            v = small.degree(node)
                            CAA_value = CAA_value + 1.0 * v / math.log(nodedegree, 2)
                        caa.append((CAA_value,{"A":node1, "B":node2}))
                        
                        # CRA
                        CRA_value = 0
                        for node in cnlist:
                            nodedegree = core.degree(node)
                            v = small.degree(node)
                            CRA_value = CRA_value + 1.0 * v / nodedegree
                        cra.append((CRA_value,{"A":node1, "B":node2}))
                        
                        # CPA
                        CPA_value = 1.0 * ex_value * ey_value + 1.0 * ex_value * CAR_value + 1.0 * ey_value * CAR_value + 1.0 * CAR_value * CAR_value 
                        cpa.append((CPA_value,{"A":node1, "B":node2}))
                        
                        
                        # 边聚类系数 p
                        ppp = CN_value*(CN_value-1)/2
                        p = 1.0*lcl/ppp

                        # Linus-CAR
                        du = 0
                        if r > 0:
                            du = 1.0/math.log(k1 * k2, 2)
                        else:
                            du = 1.0*math.log(k1 * k2, 2)
                     
                        small_nodes=small.nodes()
                        numnum=0
                        for i in xrange(len(small_nodes)):
                            node11 = small_nodes[i]
                            for j in xrange(i, len(small_nodes)):
                                node22 = small_nodes[j]
                                if i == j:
                                    continue
                                if net.has_path(small, node11, node22):
                                    numnum=numnum+1.0/net.shortest_path_length(small, node11, node22)
                        ge = 1.0*numnum/ppp
                        lcc = 1.0*p*ge

                        val = 1.0 * CN_value * lcl * lcc * du
                        linuscar.append((val,{"A":node1, "B":node2}))
                        
                        # Linus-CAA
                        LCAA_value = 1.0 * CAA_value * lcc * du
                        lcaa.append((LCAA_value,{"A":node1, "B":node2}))
                        
                        # Linus-CRA
                        LCRA_value = 1.0 * CRA_value * lcc * du
                        lcra.append((LCRA_value,{"A":node1, "B":node2}))
                        
                        # Linus-CJC 
                        LCJC_value = 1.0 * CJC_value * lcc * du
                        lcjc.append((LCJC_value,{"A":node1, "B":node2}))
                        
                        # Linus-CPA
                        LCPA_value = 1.0 * CPA_value * lcc * du
                        lcpa.append((LCPA_value,{"A":node1, "B":node2}))
                        
            # 找出命中的节点列表
            bingo_random_predictors = interlist(random_predictors, fal)
            
            th = TopkHeap(wanted)
            for i in pa:
                th.Push(i)
            trupa = refine(th.TopK())
            bingo_pa = interlist(trupa, fal)

            th = TopkHeap(wanted)
            for i in cn:
                th.Push(i)
            trucn = refine(th.TopK())
            bingo_cn = interlist(trucn, fal)
            
            th = TopkHeap(wanted)
            for i in aa:
                th.Push(i)
            truaa = refine(th.TopK())
            bingo_aa = interlist(truaa, fal)
            
            th = TopkHeap(wanted)
            for i in ra:
                th.Push(i)
            trura = refine(th.TopK())
            bingo_ra = interlist(trura, fal)
            
            th = TopkHeap(wanted)
            for i in car:
                th.Push(i)
            trucar = refine(th.TopK())
            bingo_car = interlist(trucar, fal)
     
            th = TopkHeap(wanted)
            for i in linuscar:
                th.Push(i)
            trulcar = refine(th.TopK())
            bingo_lcar = interlist(trulcar, fal)
            
            th = TopkHeap(wanted)
            for i in caa:
                th.Push(i)
            trucaa = refine(th.TopK())
            bingo_caa = interlist(trucaa, fal)
            
            th = TopkHeap(wanted)
            for i in cra:
                th.Push(i)
            trucra = refine(th.TopK())
            bingo_cra = interlist(trucra, fal)
            
            th = TopkHeap(wanted)
            for i in cjc:
                th.Push(i)
            trucjc = refine(th.TopK())
            bingo_cjc = interlist(trucjc, fal)
            
            th = TopkHeap(wanted)
            for i in cpa:
                th.Push(i)
            trucpa = refine(th.TopK())
            bingo_cpa = interlist(trucpa, fal)
            
            th = TopkHeap(wanted)
            for i in lcaa:
                th.Push(i)
            trulcaa = refine(th.TopK())
            bingo_lcaa = interlist(trulcaa, fal)
            
            th = TopkHeap(wanted)
            for i in lcra:
                th.Push(i)
            trulcra = refine(th.TopK())
            bingo_lcra = interlist(trulcra, fal)
            
            th = TopkHeap(wanted)
            for i in lcjc:
                th.Push(i)
            trulcjc = refine(th.TopK())
            bingo_lcjc = interlist(trulcjc, fal)
            
            th = TopkHeap(wanted)
            for i in lcpa:
                th.Push(i)
            trulcpa = refine(th.TopK())
            bingo_lcpa = interlist(trulcpa, fal)
            
            
            # 计算命中率
            rate_random_predictors = 1.0 * len(bingo_random_predictors) / wanted
            print "precision random_predictors : %f" % rate_random_predictors
            rate_pa = 1.0 * len(bingo_pa) / wanted
            print "precision pa : %f" % rate_pa
            rate_cn = 1.0 * len(bingo_cn) / wanted
            print "precision cn : %f" % rate_cn
            rate_ra = 1.0 * len(bingo_ra) / wanted
            print "precision ra : %f" % rate_ra
            rate_aa = 1.0 * len(bingo_aa) / wanted
            print "precision aa : %f" % rate_aa
            rate_car = 1.0 * len(bingo_car) / wanted
            print "precision car : %f" % rate_car
            rate_lcar = 1.0 * len(bingo_lcar) / wanted
            print "precision lcar : %f" % rate_lcar
            rate_caa = 1.0 * len(bingo_caa) / wanted
            print "precision caa : %f" % rate_caa
            rate_cra = 1.0 * len(bingo_cra) / wanted
            print "precision cra : %f" % rate_cra
            rate_cjc = 1.0 * len(bingo_cjc) / wanted
            print "precision cjc : %f" % rate_cjc
            
            rate_cpa = 1.0 * len(bingo_cpa) / wanted
            
            
            rate_lcaa = 1.0 * len(bingo_lcaa) / wanted
            rate_lcra = 1.0 * len(bingo_lcra) / wanted
            rate_lcjc = 1.0 * len(bingo_lcjc) / wanted
            rate_lcpa = 1.0 * len(bingo_lcpa) / wanted
        
            
            res_lcaa[cou/2-1]=res_lcaa[cou/2-1]+rate_lcaa
            res_lcra[cou/2-1]=res_lcra[cou/2-1]+rate_lcra
            res_lcjc[cou/2-1]=res_lcjc[cou/2-1]+rate_lcjc
            res_lcpa[cou/2-1]=res_lcpa[cou/2-1]+rate_lcpa
            
            res_random_predictors[cou/2-1]=res_random_predictors[cou/2-1]+rate_random_predictors
            res_pa[cou/2-1]=res_pa[cou/2-1]+rate_pa
            res_cn[cou/2-1]=res_cn[cou/2-1]+rate_cn
            res_ra[cou/2-1]=res_ra[cou/2-1]+rate_ra
            res_aa[cou/2-1]=res_aa[cou/2-1]+rate_aa
            res_car[cou/2-1]=res_car[cou/2-1]+rate_car
            res_lcar[cou/2-1]=res_lcar[cou/2-1]+rate_lcar
            res_caa[cou/2-1]=res_caa[cou/2-1]+rate_caa
            res_cra[cou/2-1]=res_cra[cou/2-1]+rate_cra
            res_cjc[cou/2-1]=res_cjc[cou/2-1]+rate_cjc
            res_cpa[cou/2-1]=res_cpa[cou/2-1]+rate_cpa
            
 
            print res_random_predictors
            print res_cn
            print res_aa
            print res_caa
            print res_ra
            print res_cra
            print res_car
            print res_lcar
            print "-----"
    after = time.time()
    print "time : %f" % (after - before)
    
    
    
    res_lcaa=func(res_lcaa,repeat)
    res_lcra=func(res_lcra,repeat)
    res_lcjc=func(res_lcjc,repeat)
    res_lcpa=func(res_lcpa,repeat)

    res_random_predictors=func(res_random_predictors,repeat)

    res_cn=func(res_cn,repeat)
    res_pa=func(res_pa,repeat)
    res_ra=func(res_ra,repeat)
    res_aa=func(res_aa,repeat)

    res_car=func(res_car,repeat)
    res_lcar=func(res_lcar,repeat)
    res_caa=func(res_caa,repeat)
    res_cra=func(res_cra,repeat)
    res_cjc=func(res_cjc,repeat)
    res_cpa=func(res_cpa,repeat)

    
    plot.figure()
    x = baifenbi
    plot.plot(x,res_pa,"b+-",label="PA",linewidth=2)
    plot.plot(x,res_cn,"gs-",label="CN",linewidth=2)
    plot.plot(x,res_ra,"bv-",label="RA",linewidth=2)
    plot.plot(x,res_aa,"c^-",label="AA",linewidth=2)
    
    plot.plot(x,res_cpa,"#4B0082",label="CPA",linewidth=2)
    plot.plot(x,res_car,"yo-",label="CAR",linewidth=2)
    plot.plot(x,res_cra,"#006400",label="CRA",linewidth=2)
    plot.plot(x,res_caa,"m+-",label="CAA",linewidth=2)
    plot.plot(x,res_cjc,"#D3D3D3",label="CJC",linewidth=2)
    
    plot.plot(x,res_lcpa,"#BDB76B",label="LCPA",linewidth=2)
    plot.plot(x,res_lcar,"rx-",label="LCAR",linewidth=2)
    plot.plot(x,res_lcra,"#8B4513",label="LCRA",linewidth=2)
    plot.plot(x,res_lcaa,"#008B8B",label="LCAA",linewidth=2)
    plot.plot(x,res_lcjc,"#00FF00",label="LCJC",linewidth=2)
    
    plot.plot(x,res_random_predictors,"k-",label="random",linewidth=2)

    
    plot.xlabel("%")
    plot.ylabel("precision")
    plot.grid(True)
    plot.legend(loc='lower right')
    if db_table_name == 'karate':
        plot.title("Zachary's karate club")
        plot.savefig(savePicName('karate', repeat))
    elif db_table_name == 'jazz':
        plot.title("Jazz musicians network")
        plot.savefig(savePicName('jazz', repeat))
    elif db_table_name == 'football':
        plot.title("American College football")
        plot.savefig(savePicName('football', repeat))
    elif db_table_name == 'food':
        plot.title("Florida Bay Food Web")
        plot.savefig(savePicName('food', repeat))
    else:
        plot.title(db_table_name)
        plot.savefig(savePicName(db_table_name, repeat))

    plot.show()
    
if __name__ == "__main__":
    db_table_name = 'Zebra'
    repeat = 1000
    process(db_table_name, repeat)
