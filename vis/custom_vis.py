import sys
import numpy as np
import time
import matplotlib.cm
import matplotlib.colors
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc, roc_auc_score
from visdom import Visdom
from typing import Callable

from utils.torch_utils import Checkpoint


mod = sys.modules[__name__]
norm = matplotlib.colors.Normalize(vmin=0, vmax=18)
cmap = matplotlib.cm.get_cmap('tab20')


class Custom_Vis:

    def __init__(self):

        self.vis = Visdom(username='', password='')
        # self.vis.env = env_name
        # self.vis.close()
        # self.pred_error = {}

    def _clear_env(self, clear: bool, env: str):
        if clear:
            self.vis.close(env=env)

    def loss_plot(self, env: str, checkpoint: Checkpoint):
        """
        The function to monitoring training.
        """
        opts = dict(title='Loss curve',
                    legend=['train loss', 'valid_loss'],
                    showlegend=True,
                    width=400,
                    height=400,
                    linecolor=np.array([[0, 0, 255], [255, 0, 0]]))
        win = np.inf
        epoch_list = checkpoint.epoch_list
        train_loss_list_per_epoch = checkpoint.train_loss_list_per_epoch
        valid_loss_list = checkpoint.valid_loss_list

        try:
            if len(epoch_list) < win:
                self._update_loss_plot(
                        epoch_list=epoch_list,
                        train_loss_list_per_epoch=train_loss_list_per_epoch,
                        valid_loss_list=valid_loss_list,
                        opts=opts,
                        win=self.loss_plt,
                        env=env)
            else:
                self._update_loss_plot(
                    epoch_list=epoch_list[-win:],
                    train_loss_list_per_epoch=train_loss_list_per_epoch[-win:],
                    valid_loss_list=valid_loss_list[-win:],
                    opts=opts,
                    win=self.loss_plt,
                    env=env)
        except AttributeError:
            self.loss_plt = self.vis.line(
                    X=np.array([epoch_list, epoch_list]).T,
                    Y=np.array([train_loss_list_per_epoch, valid_loss_list]).T,
                    opts=opts,
                    env=env)

    def print_params(self, env: str, params: dict, title: str,
                     clear: bool = False):
        """
        Print function to display parameters
        """
        self._clear_env(clear, env)
        opts = dict(width=400, height=400)
        text = f'<h2> {title} </h2><br>'

        for keys, values in params.items():
            text += f'<h4>{keys}: {values} </h4>'

        self.vis.text(text, opts=opts, env=env)

    def data_plot(self, env: str, data: np.ndarray, ano_set: np.ndarray,
                  ano_features: np.ndarray, clear: bool = True):
        """
        The function for plot of anomaly subsequence.
        """
        self._clear_env(clear, env)
        if len(ano_set) > 10:
            size = 10
        else:
            size = len(ano_set)

        opts = dict(width=1200, height=400)
        selected = np.random.choice(a=np.arange(len(ano_set)),
                                    size=size,
                                    replace=False)
        for sel in selected:
            setattr(self, f'v_{sel}', self.vis.line(X=[0, 1],
                                                    Y=[0, 1],
                                                    opts=opts,
                                                    env=env))
            fig = go.Figure()
            ano_idx, feature = ano_set[sel], ano_features[sel]
            s, e = ano_idx[0], ano_idx[-1]
            margin = e-s
            for f in feature:
                fig.add_trace(go.Scatter(x=np.arange(s-margin, e+margin),
                                         y=data[s-margin:e+margin, f],
                                         mode='lines',
                                         name=str(f)))
            fig.update_layout(shapes=[dict(type='rect',
                                           xref='x',
                                           yref='paper',
                                           x0=s,
                                           y0=0,
                                           x1=e,
                                           y1=1,
                                           fillcolor='LightSalmon',
                                           opacity=0.5,
                                           layer='below',
                                           line_width=0)],
                              title='Comparision between normal and anomaly',
                              title_x=0.5,
                              autosize=False,
                              width=900,
                              height=400,
                              xaxis=go.layout.XAxis(title='timestamp',
                                                    showticklabels=False))
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)
            self.vis.plotlyplot(fig, win=getattr(self, f'v_{sel}'), env=env)

    def print_training(self, env: str, EPOCH: str, epoch: str,
                       training_time: float, iter_time: float,
                       avg_train_loss: float, valid_loss: float,
                       patience: int, counter: int):
        """
        Print training status and loss.
        p: patience
        c: counter
        """
        t_m = f'{int(training_time//60):2d}'
        t_s = f'{training_time%60:5.2f}'
        i_m = f'{int(iter_time//60):2d}'
        i_s = f'{iter_time%60:5.2f}'
        state = patience - counter
        opts = dict(width=400, height=400)

        if patience != np.inf:
            text = '<h2> Training status </h2><br>'\
                f'\r <h4>Epoch: {epoch:3d} / {str(EPOCH):3s}</h4>'\
                f'<h4>train time: {t_m}m {t_s}s</h4>'\
                f'<h4>iteration time: {i_m}m {i_s}s</h4>'\
                f'<h4>avg train loss: {avg_train_loss}</h4>'\
                f'<h4>valid loss: {valid_loss}</h4>'\
                f'\r <h4>EarlyStopping: {">"*counter+"-"*(state)} |</h4>'

        else:
            text = '<h2> Training status </h2><br>'\
                f'\r <h4>Epoch: {epoch:3d} / {str(EPOCH):3s}</h4>'\
                f'<h4>train time: {t_m}m {t_s}s</h4>'\
                f'<h4>iteration time: {i_m}m {i_s}s</h4>'\
                f'<h4>avg train loss: {avg_train_loss}</h4>'\
                f'<h4>valid loss: {valid_loss}</h4>'

        try:
            self.vis.text(text,
                          win=self.training,
                          append=False,
                          opts=opts,
                          env=env)
        except AttributeError:
            self.training = self.vis.text(text, opts=opts, env=env)

    def score_distribution(self, env: str, anomaly_label: np.ndarray,
                           anomaly_score: np.ndarray,
                           filtered_score: np.ndarray):
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(x=anomaly_score[anomaly_label == 0],
                                         histnorm='probability density',
                                         name='normal'))
        fig_score.add_trace(go.Histogram(x=anomaly_score[anomaly_label == 1],
                                         histnorm='probability density',
                                         name='anomaly'))

        if not isinstance(filtered_score, type(None)):
            fig_score.add_trace(go.Histogram(x=filtered_score,
                                             histnorm='probability density',
                                             name='filtered'))

        fig_score.update_layout(title_text='Anomaly Score Distribution',
                                title_x=0.5,
                                barmode='overlay',
                                xaxis=go.layout.XAxis(title='anomaly score'),
                                autosize=False,
                                width=600,
                                height=300)
        fig_score.update_traces(opacity=0.5)

        try:
            self.vis.plotlyplot(fig_score, win=self.vis_score, env=env)
        except AttributeError:
            self.vis_score = self.vis.plotlyplot(fig_score, env=env)

    def ROC_curve(self, env: str, anomaly_label: np.ndarray,
                  anomaly_score: np.ndarray):
        label = anomaly_label
        score = anomaly_score
        if np.unique(label).shape[0] == 1:
            return
        fpr, tpr, thresholds = roc_curve(label, score)
        auroc = roc_auc_score(label, score)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr,
                                     y=tpr,
                                     mode='lines',
                                     fill='tozeroy',
                                     name='ROC'))
        fig_roc.add_shape(type='line',
                          line=dict(dash='dash'),
                          x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_layout(
                title_text=f'ROC Curve({auroc:.4f})',
                title_x=0.5,
                xaxis=go.layout.XAxis(title='False Positive Rate'),
                yaxis=go.layout.YAxis(title='True Positive Rate'),
                autosize=False,
                width=600,
                height=600)

        # insert to visdom
        try:
            self.vis.plotlyplot(fig_roc, win=self.vis_roc, env=env)
        except AttributeError:
            self.vis_roc = self.vis.plotlyplot(fig_roc, env=env)

        return auroc

    def _update_loss_plot(self, epoch_list: list,
                          train_loss_list_per_epoch: list,
                          valid_loss_list: list, opts: dict, win: Callable,
                          env: str):
        self.vis.line(
                X=np.array([epoch_list, epoch_list]).T,
                Y=np.array([train_loss_list_per_epoch, valid_loss_list]).T,
                opts=opts,
                win=win,
                update='replace',
                env=env)

    def lof_score(self, env: str, lof_score: np.ndarray):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=lof_score,
                                   histnorm='probability density',
                                   name='lof_score'))
        fig.update_layout(title_text='LOF Score Distribution',
                          title_x=0.5,
                          barmode='overlay',
                          autosize=False,
                          width=600,
                          height=600)
        fig.update_traces(opacity=0.5)
        self.vis.plotlyplot(fig, env=env)
