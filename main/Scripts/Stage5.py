import numpy as np
import time
import pandas as pd
import math
import matplotlib.pyplot as plt

def Stage5(config):
    new_symbolic_list=np.load('New_symbolic_list1.npy')
    final=np.load('Selected_Hardcoded_directionalmatrix1.npy',allow_pickle = True )
    formula=config['Consistency_formula']
    
    time_start = time.time()
    north_table = []
    south_table=[]
    east_table=[]
    west_table=[]
    for num in range(1000):
        trial=new_symbolic_list[num]
        for poi in range(len(final)):
            for ele in range(len(final[poi])):
                data=final[poi][ele]
                a_id=trial[data[1]][data[0]]
                b_id=trial[data[3]][data[2]]
                dictionary_data_north = {'a_id': a_id, 'b_id' : b_id, 'count_north': data[4][0]}
                dictionary_data_south= {'a_id': a_id, 'b_id' : b_id, 'count_south': data[4][1]}
                dictionary_data_east= {'a_id': a_id, 'b_id' : b_id, 'count_east': data[4][2]}
                dictionary_data_west= {'a_id': a_id, 'b_id' : b_id, 'count_west': data[4][3]}          
                north_table.append(dictionary_data_north)
                south_table.append(dictionary_data_south)
                east_table.append(dictionary_data_east)
                west_table.append(dictionary_data_west)
                    
    north_table = pd.DataFrame.from_dict(north_table)
    north_table = north_table[(north_table[['count_north']] != 0).all(axis=1)]
    north_table = pd.pivot_table(north_table, values='count_north', index=['a_id', 'b_id'], aggfunc=np.sum)
    north_table = north_table.sort_values(('count_north'), ascending=False)
    
    south_table = pd.DataFrame.from_dict(south_table)
    south_table = south_table[(south_table[['count_south']] != 0).all(axis=1)]
    south_table = pd.pivot_table(south_table, values='count_south', index=['a_id', 'b_id'],aggfunc=np.sum)
    south_table = south_table.sort_values(('count_south'), ascending=False)
    
    east_table = pd.DataFrame.from_dict(east_table)
    east_table = east_table[(east_table[['count_east']] != 0).all(axis=1)]
    east_table = pd.pivot_table(east_table, values='count_east', index=['a_id', 'b_id'], aggfunc=np.sum)
    east_table = east_table.sort_values(('count_east'), ascending=False)
    
    west_table = pd.DataFrame.from_dict(west_table)
    west_table = west_table[(west_table[['count_west']] != 0).all(axis=1)]
    west_table = pd.pivot_table(west_table, values='count_west', index=['a_id', 'b_id'], aggfunc=np.sum)
    west_table = west_table.sort_values(('count_west'), ascending=False)
    
    print("Co Occurrence Tables Created. Time elapsed: {} seconds".format(time.time()-time_start))
    
    north_table.to_csv('north_table1.csv') #Remove 2 for orignal
    south_table.to_csv('south_table1.csv') 
    east_table.to_csv('east_table1.csv') 
    west_table.to_csv('west_table1.csv') 
    
    north_table = pd.read_csv("north_table1.csv") 
    south_table = pd.read_csv("south_table1.csv") 
    east_table = pd.read_csv("east_table1.csv") 
    west_table = pd.read_csv("west_table1.csv") 
    
    if formula=='NPMI':
        time_start = time.time()
        north_table['north_consistency'] = 0 
        total_count_north = north_table['count_north'].sum()
        for i, j in north_table.iterrows(): 
            cooc_probability= north_table.loc[(north_table['a_id'] == j[0]) & (north_table['b_id'] == j[1]),'count_north'].sum() / total_count_north
            marginal_a_id_probability= north_table.loc[north_table['a_id'] == j[0], 'count_north'].sum() / total_count_north
            marginal_b_id_probability= north_table.loc[north_table['b_id'] == j[1], 'count_north'].sum() /total_count_north
            consistency_north=float(np.log2(cooc_probability / (marginal_a_id_probability * marginal_b_id_probability))/np.log2(1 / cooc_probability))
            north_table.loc[(north_table['a_id'] == j[0]) & (north_table['b_id'] == j[1]), 'north_consistency'] = consistency_north
        
        south_table['south_consistency'] = 0 
        total_count_south = south_table['count_south'].sum()
        for i, j in south_table.iterrows(): 
            cooc_probability= south_table.loc[(south_table['a_id'] == j[0]) & (south_table['b_id'] == j[1]),'count_south'].sum() / total_count_south
            marginal_a_id_probability= south_table.loc[south_table['a_id'] == j[0], 'count_south'].sum() / total_count_south
            marginal_b_id_probability= south_table.loc[south_table['b_id'] == j[1], 'count_south'].sum() /total_count_south
            consistency_south=float(np.log2(cooc_probability / (marginal_a_id_probability * marginal_b_id_probability))/np.log2(1 / cooc_probability))
            south_table.loc[(south_table['a_id'] == j[0]) & (south_table['b_id'] == j[1]), 'south_consistency'] = consistency_south
            
        east_table['east_consistency'] = 0 
        total_count_east = east_table['count_east'].sum()
        for i, j in east_table.iterrows(): 
            cooc_probability= east_table.loc[(east_table['a_id'] == j[0]) & (east_table['b_id'] == j[1]),'count_east'].sum() / total_count_east
            marginal_a_id_probability= east_table.loc[east_table['a_id'] == j[0], 'count_east'].sum() / total_count_east
            marginal_b_id_probability= east_table.loc[east_table['b_id'] == j[1], 'count_east'].sum() /total_count_east
            consistency_east=float(np.log2(cooc_probability / (marginal_a_id_probability * marginal_b_id_probability))/np.log2(1 / cooc_probability))
            east_table.loc[(east_table['a_id'] == j[0]) & (east_table['b_id'] == j[1]), 'east_consistency'] = consistency_east
        
        west_table['west_consistency'] = 0 
        total_count_west = west_table['count_west'].sum()
        for i, j in west_table.iterrows(): 
            cooc_probability= west_table.loc[(west_table['a_id'] == j[0]) & (west_table['b_id'] == j[1]),'count_west'].sum() / total_count_west
            marginal_a_id_probability= west_table.loc[west_table['a_id'] == j[0], 'count_west'].sum() / total_count_west
            marginal_b_id_probability= west_table.loc[west_table['b_id'] == j[1], 'count_west'].sum() /total_count_west
            consistency_west=float(np.log2(cooc_probability / (marginal_a_id_probability * marginal_b_id_probability))/np.log2(1 / cooc_probability))
            west_table.loc[(west_table['a_id'] == j[0]) & (west_table['b_id'] == j[1]), 'west_consistency'] = consistency_west
        
        print("Consistencies added using NPMI formula. Time elapsed: {} seconds".format(time.time()-time_start))
    
    else:
        time_start = time.time()
        north_table['north_consistency'] = 0 
        for i, j in north_table.iterrows(): 
            plr=north_table.loc[north_table['a_id'] == j[0], 'count_north'].sum()
            pll=north_table.loc[north_table['b_id'] == j[1], 'count_north'].sum()
            pt=north_table.loc[(north_table['a_id'] == j[0]) & (north_table['b_id'] == j[1]),'count_north'].sum()
            consistency=math.log(pt/(plr*pll))
            north_table.loc[(north_table['a_id'] == j[0]) & (north_table['b_id'] == j[1]), 'north_consistency'] = consistency
        
        south_table['south_consistency'] = 0 
        for i, j in south_table.iterrows(): 
            plr=south_table.loc[south_table['a_id'] == j[0], 'count_south'].sum()
            pll=south_table.loc[south_table['b_id'] == j[1], 'count_south'].sum()
            pt=south_table.loc[(south_table['a_id'] == j[0]) & (south_table['b_id'] == j[1]),'count_south'].sum()
            consistency=math.log(pt/(plr*pll))
            south_table.loc[(south_table['a_id'] == j[0]) & (south_table['b_id'] == j[1]), 'south_consistency'] = consistency
        
        east_table['east_consistency'] = 0 
        for i, j in east_table.iterrows(): 
            plr=east_table.loc[east_table['a_id'] == j[0], 'count_east'].sum()
            pll=east_table.loc[east_table['b_id'] == j[1], 'count_east'].sum()
            pt=east_table.loc[(east_table['a_id'] == j[0]) & (east_table['b_id'] == j[1]),'count_east'].sum()
            consistency=math.log(pt/(plr*pll))
            east_table.loc[(east_table['a_id'] == j[0]) & (east_table['b_id'] == j[1]), 'east_consistency'] = consistency
        
        west_table['west_consistency'] = 0 
        for i, j in west_table.iterrows(): 
            plr=west_table.loc[west_table['a_id'] == j[0], 'count_west'].sum()
            pll=west_table.loc[west_table['b_id'] == j[1], 'count_west'].sum()
            pt=west_table.loc[(west_table['a_id'] == j[0]) & (west_table['b_id'] == j[1]),'count_west'].sum()
            consistency=math.log(pt/(plr*pll))
            west_table.loc[(west_table['a_id'] == j[0]) & (west_table['b_id'] == j[1]), 'west_consistency'] = consistency
        
        print("Consistencies added using NORMAL Formula. Time elapsed: {} seconds".format(time.time()-time_start))    
        
    plt.show(north_table["north_consistency"].plot.hist(bins=100))
    
    north_table.to_csv('north_table1.csv', index=False) #Remove 2 for orignal
    south_table.to_csv('south_table1.csv', index=False) 
    east_table.to_csv('east_table1.csv', index=False) 
    west_table.to_csv('west_table1.csv', index=False) 
    print("Stage 5 Completed Succesfully!")