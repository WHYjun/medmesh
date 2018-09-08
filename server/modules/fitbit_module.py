import datetime
import fitbit
import gather_keys_oauth2 as Oauth2
import os

FITBIT_ID = os.environ.get('FITBIT_ID')
FITBIT_SECRET = os.environ.get('FITBIT_SECRET')


def get_token():
    server = Oauth2.OAuth2Server(
        FITBIT_ID, FITBIT_SECRET, redirect_uri='http://127.0.0.1:8080/')
    server.browser_authorize()
    access_token = str(server.fitbit.client.session.token['access_token'])
    refresh_token = str(server.fitbit.client.session.token['refresh_token'])
    print('Update your access token to {}'.format(access_token))
    print('Update your refresh token to {}'.format(refresh_token))


def get_heartrate():
    access_token = os.environ.get('ACCESS_TOKEN')
    refresh_token = os.environ.get('REFRESH_TOKEN')
    auth2_client = fitbit.Fitbit(FITBIT_ID, FITBIT_SECRET,
                                 oauth2=True,
                                 access_token=access_token,
                                 refresh_token=refresh_token)
    yesterday = str((datetime.datetime.now() -
                     datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    fit_statsHR = auth2_client.intraday_time_series(
        'activities/heart', base_date=yesterday, detail_level='1sec')
    print(fit_statsHR)


if __name__ == '__main__':
    get_heartrate()
