'''
This script contains the report generationg commands for
* Validation report:
    The report that provides a spreadsheet allowing the user to review
    results images from the analysis and denote their validity
* Collation:
    Collects, trims and prunes data from the anaylyses output files.
    collects: finds the analysis csv files for a given type
    trims: removes the invalid objects according to the validation report
    prunes: Takes only the desired fields
        Desired fields:
            1. 100 measurements from the width evenly dispersed
            2. filName, objectNumber, colorIndex, averageWidth, maxWidth, length 

'''
import numpy as np
import click
import pandas as pd
import os
import os.path as osp
import logging
from glob import glob
import json
import subprocess as sp

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)

@click.group()
@click.argument(
    'analysis_type',
    type=click.Choice(['ear', 'cob', 'kernel'])
)
@click.pass_context
def report(ctx, analysis_type):
    ''' Report generation '''
    ctx.obj['analysis_type'] = analysis_type
    return

'''
I removed the clear option to avoid deleting completed work.
If someone wants to restart the validation report, they will have to manually delete two files:
    1. report.hist
    2. val_report.csv
Then regenerate the collated report afterwards
'''

@report.command()
@click.pass_context
def generate(ctx):
    ''' Generate either a validation or summary report '''
    path = osp.join(ctx.obj.get('local_base'), 'results', ctx.obj.get('analysis_type'))

    val_path = osp.join(path, 'val_report.csv')
    if osp.exists(val_path):
        report = pd.read_csv(val_path)
        columns = report.columns
    else:
        if ctx.obj.get('analysis_type') in ['cob', 'ear']:
            columns = ['sample', 'group', '1', '2', '3', 'notes', 'dir']
        elif ctx.obj.get('analysis_type') in ['kernel']:
            columns = ['sample', 'group', 'valid', 'notes', 'dir']
        report = pd.DataFrame(columns=columns)

    hist_path = osp.join(path, 'report.hist')
    if osp.exists(hist_path):
        with open(hist_path, 'r') as hist_file:
            hist = json.load(hist_file)
    else:
        hist = []

    log.info(f"Searching {path} for analyses") 
    for result_dir in os.listdir(path):
        result_path = osp.join(path, result_dir)
        if not osp.isdir(result_path): 
            continue
        if result_dir in hist:
            log.info(f'Already in validation report, skipping: {result_dir}')
            continue
        csvs = glob(osp.join(result_path, 'output', '*.csv'))
        if len(csvs) < 1:
            log.warning(f'{result_dir} does not have a result file, skipping.')
            continue
        elif len(csvs) > 1:
            output_dir = osp.join(result_path, 'output')
            log.error(f'{result_dir} has more than one result file,' \
                + f' please delete unecessary files in {output_dir}' \
                + ' and re-run this command.')
            quit()
        else:
            result_file = csvs.pop()
        
        log.info(f"Adding to report: {result_dir}")
        data = pd.read_csv(result_file)

        filenames = data['fileName'].unique()
        try:
            name  = [filename.split('_')[0] for filename in filenames]
            group = [filename.split('_')[1] for filename in filenames]
        except IndexError as ie:
            log.error(ie)
            log.error("Invalid Filenames based on length")
            for filename in filenames:
                if len(filename.split('_')) == 1:
                    log.error(filename)
            quit()

        curr_report = pd.DataFrame(columns=columns)
        curr_report['sample'] = name
        curr_report['group'] = group
        curr_report['dir'] = result_dir

        report = report.append(curr_report)
        hist.append(result_dir)

        unique_samples = pd.DataFrame(report['sample'].unique())
        unique_samples.to_csv(osp.join(path, 'analyzed_samples.csv'), index=False)

    if ctx.obj.get('analysis_type') in ['cob', 'ear']:
        report = report.sort_values(by=['sample', 'orientation'])
    elif ctx.obj.get('analysis_type') in ['kernel']:
        report = report.sort_values(by=['sample', 'group'])

    log.info("Validation report preview")
    log.info(report.head())
    report.to_csv(val_path, index=False)
    with open(hist_path, 'w') as hist_file:
        json.dump(hist, hist_file)

    return


@report.command()
@click.pass_context
def collate(ctx):
    ''' Process results files and collate '''
    results_dir = osp.join(
        ctx.obj.get('local_base'),
        'results',
        ctx.obj.get('analysis_type')
    )

    sort_columns = ['row', 'group', 'analysis_name']

    if ctx.obj.get('analysis_type') in ['ear', 'cob']:
        results_columns = ['fileName', 'objectNumber', 'default__1_maxWidth__1',
        'default__1_averageWidth__1', 'default__1_length__1']

        if ctx.obj.get('analysis_type') == 'ear':
            results_columns += ['default__1_kernelLength__1']

        results_columns += \
            ['along__1_position__' + str(i) for i in list(range(1, 1000, 10))]

    elif ctx.obj.get('analysis_type') == 'kernel':
        results_columns = ['fileName', 'objectNumber', 'default__1_majorAxis__1',
            'default__1_minorAxis__1', 'default__1_area__1']

        sort_columns.insert(2, 'objectNumber') 

    cumulative = pd.DataFrame(columns=results_columns, dtype=int)
    cumulative['fileName'] = cumulative['fileName'].astype(str)
    
    analysis_names_col = pd.Series()
    for results_file in glob(osp.join(results_dir, '*', 'output', '*.csv')):
        log.info("Appending to cumulative dataframe: " + osp.basename(results_file))
        current_results_df = pd.read_csv(results_file)
        cumulative = cumulative.append(current_results_df[cumulative.columns])
        analysis_name = osp.basename(osp.dirname(osp.dirname(results_file)))
        analysis_names_col = analysis_names_col.append(
            pd.Series(np.full(len(current_results_df), analysis_name)))

    results_columns += ['analysis_name']
    cumulative['analysis_name'] = analysis_names_col

    cumulative['row'] = cumulative['fileName'].apply(lambda x: str(x.split("_")[0]))
    cumulative['group'] = cumulative['fileName'].apply(lambda x: str(x.split("_")[1]))

    results_columns = ['row', 'group'] + results_columns

    log.info("Dropping duplicate rows")
    nrows = cumulative.shape[0]
    duplicate_removal_columns = list(cumulative.columns)
    duplicate_removal_columns.remove('analysis_name')
    cumulative = cumulative.drop_duplicates(subset=duplicate_removal_columns)
    ndropped = nrows - cumulative.shape[0]
    log.info("Number of rows dropped: " + str(ndropped))

    log.info("Convert pixels to millimeters")
    if ctx.obj.get('analysis_type') in ['ear', 'kernel']:
        dpmm = 47.24409 # dots per millimeter (like dpi)
    elif ctx.obj.get('analysis_type') in ['cob']:
        dpmm = 11.81102  # dots per millimeter (like dpi)
    else:
        raise Exception

    for column in results_columns:
        if column not in ['fileName', 'row', 'group', 'analysis_name', 'objectNumber']:
            if column == 'default__1_area__1':
                cumulative[column] = cumulative[column].apply(lambda x: x / (dpmm ** 2))
            else:
                cumulative[column] = cumulative[column].apply(lambda x: x / dpmm)

    log.info('Sorting by row number')
    cumulative = cumulative.sort_values(by=sort_columns)

    # CLEAN INVALID ENTRIES
    log.info('Cleaning invalid entries based on validation report')
    validation_data = pd.read_csv(osp.join(results_dir, 'val_report.csv'))

    if ctx.obj.get('analysis_type') in ['cob', 'ear']:
        # check if the validation report is completed
        if validation_data[['1', '2', '3']].isna().sum().sum() > 0:
            log.error(f"Please complete the {ctx.obj.get('analysis_type')} " \
                + "validation report before running this command")
            return

        for _, row in validation_data.iterrows():
            row_num = str(row['sample'])
            group = '00' + str(row['group'])
            for i in range(1,4):
                if row[str(i)] == -1:
                    cumulative = cumulative[
                        ~((cumulative['row'] == row_num)
                        & (cumulative['group'] == group)
                        & (cumulative['objectNumber'] == i))
                    ]
                elif (ctx.obj.get('analysis_type') == 'ear') \
                    and (row[str(i)] == 0):
                    cumulative.loc[((cumulative['row'] == row_num)
                                 & (cumulative['group'] == group)
                                 & (cumulative['objectNumber'] == i)),
                                'default__1_kernelLength__1'] = np.nan
    elif ctx.obj.get('analysis_type') == 'kernel':
        if validation_data['valid'].isna().sum() > 0:
            log.error(f"Please complete the {ctx.obj.get('analysis_type')} " \
                + "validation report before running this command")                
            return
        for _, row in validation_data.iterrows():
            row_num = str(row['sample'])
            group = '00' + str(row['group'])
            if row['valid'] != 1:
                cumulative = cumulative[
                    ~((cumulative['row'] == row_num)
                    & (cumulative['group'] == group))
                ]

    log.info("Cumulative dataframe preview of top rows:")
    log.info(cumulative[results_columns].head())

    save_loc = osp.join(
        results_dir, 
        'collated_' + ctx.obj.get('analysis_type') + '_results.csv'
    )

    log.info("Saving result to " + save_loc)
    results_columns.remove('fileName')
    cumulative[results_columns].to_csv(save_loc, index=False)

    log.info("Copying to current working directory")
    sp.run(['cp', save_loc,
        osp.join(os.getcwd(), osp.basename(save_loc))])
    
    if ctx.obj.get('analysis_type') == 'kernel':
       
        summary_report = _produce_kernel_summary_report(cumulative)
        
        log.info("Kernel summary report top rows")
        log.info(summary_report.head())
        
        save_loc = osp.join(
            results_dir, 
            'collated_' + ctx.obj.get('analysis_type') + '_summary.csv'
        )
    
        log.info("Saving result to " + save_loc)
        final_columns_ordered = [
            'row',
            'group',
            'mean_width',
            'mean_length',
            'mean_area'
        ]
        summary_report[final_columns_ordered].to_csv(save_loc, index=False)
    
        log.info("Copying to current working directory")
        sp.run(['cp', save_loc,
            osp.join(os.getcwd(), osp.basename(save_loc))])       


def _produce_kernel_summary_report(data):
    column_conversion = {
        'fileName':                 'fileName',
        'default__1_majorAxis__1':  'mean_width',
        'default__1_minorAxis__1':  'mean_length',
        'default__1_area__1':       'mean_area'
    }
    summary_report = data[column_conversion.keys()].groupby('fileName').mean().reset_index()
    summary_report['row'] = summary_report['fileName'].apply(lambda x: x.split("_")[0])
    summary_report['group'] = summary_report['fileName'].apply(lambda x: x.split("_")[1])    
    summary_report = summary_report.rename(columns=column_conversion)
    return summary_report


if __name__ == '__main__':
    report({})