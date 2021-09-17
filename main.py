import numpy as np
import pandas as pd
from smoothing import triangular_smooth
from pandas.tseries.offsets import MonthEnd
import pickle
import warnings


class SalesRegressionInfo:
    def __init__(self, fes, scan_period):
        self.fes = fes
        self.scan_period = scan_period

        # parameters initialization
        self.fes_triangle = dict()
        self.mean_fes_peak_day = dict()
        self.predictors = None
        self.responses = None

        self.no_fes_coef = 0

    def _get_triangle_list(self, sales, fes_list, fes_code=1):
        """
        gets the x axis of triangles for the given festival in each year given fes_list

        sales(pandas series): index:sales date, values:sales amount
        fes_list(pandas series): index:date, values:festival code
        fes_code(int, str): festival code to be manipulated
        scan_period(int): day for scanning for finding sales peak before festival
        return: a list of tuples: (sym_date, peak_date, fes_date), each tuple
        represents x axis of the triangle
        """
        fes_list = fes_list[fes_list == fes_code]
        triangle_list = []

        for fes_date in fes_list.index:
            scan_min = fes_date - pd.DateOffset(days=self.scan_period)
            min_fes_period = sales.loc[scan_min: fes_date]
            if not min_fes_period.empty:
                peak_date = min_fes_period.idxmax()
                sym_date = peak_date - (fes_date - peak_date)
                triangle_list.append((sym_date, peak_date, fes_date))

        return triangle_list

    def _get_ybond(self, m_bond, x, y):
        m_idx = 0
        while m_bond > x[m_idx]:
            m_idx += 1
        y_bond = np.interp(
            x=[m_bond],
            xp=[x[m_idx - 1], x[m_idx]],
            fp=[y[m_idx - 1], y[m_idx]]
        )[0]

        return y_bond, m_idx

    def _get_area_ratio(self, start, peak, end, left, right):
        y = [0, 0, 1, 0, 0]
        left = (left - peak).days
        right = (right - peak).days
        x = np.array([
            (start - pd.DateOffset(days=360) - peak).days,
            (start - peak).days,
            (peak - peak).days,
            (end - peak).days,
            (end + pd.DateOffset(days=360) - peak).days
        ])

        total_area = (x[3] - x[1]) / 2

        if total_area == 0:
            coef = 1

        else:
            start_y_bond, start_idx = self._get_ybond(left, x, y)
            end_y_bond, end_idx = self._get_ybond(right, x, y)

            x_point = [left] + [x[i] for i in range(start_idx, end_idx)] + [right]
            y_point = [start_y_bond] + [y[i] for i in range(start_idx, end_idx)] + [end_y_bond]
            area = np.trapz(
                x=x_point,
                y=y_point
            )

            coef = area / total_area

        assert 0 <= coef <= 1, 'wrong coef value range, coef={0}'.format(coef)

        return coef

    def fit_transform(self, ser, resample_param='MS'):
        """
        fit data and get the following information for sales prediction
        self.mean_fes_peak_day, self.predictors, self.responses

        *mean_fes_peak_day: dictionary
            mean day between festival and sales peak for each festival and item

        *predictors: 2D numpy array
            coef 0: year
            coef 1 ~ 3: festival influence
            coef 4 ~15: month encoding

        *responses: 1D numpy array
            array of sales

        ser: sales series data
        resample_param:
        return: None
        """
        for fes_code in self.fes.unique():
            self.fes_triangle[fes_code] = self._get_triangle_list(
                sales=ser, fes_list=self.fes, fes_code=fes_code
            )

        # get average days between festival and peak
        for fes_code in self.fes_triangle:
            mean_day = \
                np.mean([(end - peak).days for start, peak, end in self.fes_triangle[fes_code]])
            self.mean_fes_peak_day[fes_code] = mean_day

        ser_mon = ser.resample(resample_param).sum()

        # iterate over each month and festival for checking if there are associations
        # between month sale and triangle
        predictors = []
        responses = []
        for mon_start_date, mon_sale in ser_mon.iteritems():
            # set response
            responses.append(mon_sale)

            # predictor 0: sales amount
            predictor_tmp = [mon_start_date.year]

            # predictor 1 ~ 3: festival influence
            mon_end_date = mon_start_date + MonthEnd(1)
            for fes_code, triplet_list in self.fes_triangle.items():
                coef_list = []
                for start, peak, end in triplet_list:
                    if mon_start_date <= end and mon_end_date >= start:
                        coef = self._get_area_ratio(
                            start=start, peak=peak, end=end,
                            left=mon_start_date, right=mon_end_date
                        )
                        coef_list.append(coef)

                if len(coef_list) > 1:
                    warnings.warn(
                        'mon_start_date={0}, fes_code={1} has multiple years influence.'
                            .format(mon_start_date, fes_code)
                    )

                coef = np.sum(coef_list) if coef_list else self.no_fes_coef
                predictor_tmp.append(coef)

            # predictor 4 ~ 15: month encoding
            mon_encoding = [0] * 12
            mon_encoding[mon_start_date.month-1] = 1
            predictor_tmp.extend(mon_encoding)

            predictors.append(predictor_tmp)

        # to numpy array
        self.predictors = np.array(predictors)
        self.responses = np.array(responses)


if __name__ == '__main__':
    # reading data
    sales_datapath = './src_data/df_sales_barcode_target.pickle'
    fes_datapath = './src_data/festival.pickle'
    sales = pd.read_pickle(sales_datapath)
    fes = pd.read_pickle(fes_datapath)

    # parameter setting
    item_list = sales['條碼'].unique()
    smoo_shift_day = 15
    scan_period = 60

    # initialization
    true_sales = dict()
    predictors = dict()
    responses = dict()
    fes_peak_day = dict()

    # iterate over each item
    for item in item_list:
        s = sales[sales['條碼'] == item][['日期', '銷貨數量(箱)']].set_index('日期')
        true_sales[item] = s.resample('MS').sum()

        # get smooth data
        s_smoo = triangular_smooth(time_ser=s, shift_day=smoo_shift_day)

        # get predictors, responses, mean day between peak and festival
        regr_info = SalesRegressionInfo(fes=fes, scan_period=scan_period)
        regr_info.fit_transform(s_smoo)

        predictors[item] = regr_info.predictors
        responses[item] = regr_info.responses
        fes_peak_day[item] = regr_info.mean_fes_peak_day

    with open('./regression_info/predictors.pickle', 'wb') as f:
        pickle.dump(predictors, f)

    with open('./regression_info/responses.pickle', 'wb') as f:
        pickle.dump(responses, f)

    with open('./regression_info/fes_peak_day.pickle', 'wb') as f:
        pickle.dump(fes_peak_day, f)

    with open('./regression_info/true_sales.pickle', 'wb') as f:
        pickle.dump(true_sales, f)
