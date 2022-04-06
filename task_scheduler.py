#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tal Golan
"""


## Manage HPC jobs with sqlite
#
#
## A simple example (uncomment the code below to try it out)
#
## We start a new database.
## Here, we define that if a job runs for more than 6 hours it is considered lost and is restarted.
#
# import numpy as np
# sch=TaskScheduler(db_path='/home/tal/scheduler.db',max_job_time_in_seconds=6*3600)
#
# for gamma in [-1e3,1,1e5]:
#    for category in ['cats','dogs']:
#        job_id={'gamme':gamma,'category':category} # any JSONable type will do (e.g. numbers, strings, dictionaries, lists...)
#        success=sch.start_job(job_id)
#        if not success:
#            # the job already started / completed. skip to the next loop
#            continue
#
#        # computation comes here
#        results=np.random.uniform(size=(4,2))
#
#        # we are done. let's set the 'done' flag.
#        sch.job_done(job_id,results=results) # saving results is optional. they are converted to string.
#
## if we are done, read the table to pandas:
# DF=sch.to_pandas()
# print(DF)

import sqlite3
import json, os, time, math
import pandas as pd


class TaskScheduler:
    def __init__(self, db_path="scheduler.db", max_job_time_in_seconds=12 * 3600):
        self.db_path = db_path
        self.max_job_time_in_seconds = max_job_time_in_seconds

        if not os.path.isfile(self.db_path):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()

            # Create table
            c.execute(
                """CREATE TABLE jobs
                 (job_id TEXT UNIQUE, is_completed BOOLEAN, time_started INTEGER, results TEXT)"""
            )

            conn.commit()  # Save (commit) the changes
            conn.close()

    def _execute_sqlite(self, query, parameters=None):
        if parameters is None:
            parameters = ()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(query, parameters)
            conn.commit()  # Save (commit) the changes
            success = True
        except Exception as e:
            if (
                str(e) != "UNIQUE constraint failed: jobs.job_id"
            ):  # show error, unless it's a duplicate unique job_id
                print("sqlite error:", str(e))
                print(str(e))
            success = False
        finally:
            conn.close()
        return success

    def _sqlite_fetchone(self, query, parameters=None):
        if parameters is None:
            parameters = ()
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute(query, parameters)
            fetched = c.fetchone()[0]
        except Exception as e:
            print("sqlite error:", str(e))
            fetched = None
        finally:
            conn.close()
        return fetched

    def start_job(self, job_id):
        job_id_json = json.dumps(job_id)
        time_started = math.ceil(time.time())

        # delete old incompleted jobs
        self._execute_sqlite(
            "DELETE FROM jobs WHERE (job_id = ? AND time_started < ? AND is_completed = 0)",
            (job_id_json, time_started - self.max_job_time_in_seconds),
        )

        # try inserting new job
        success = self._execute_sqlite(
            "INSERT INTO jobs (job_id, is_completed, time_started) VALUES (?, 0, ?)",
            (job_id_json, time_started),
        )

        # when failed, provide feedback
        if success == False:
            is_completed = self._sqlite_fetchone(
                "SELECT is_completed FROM jobs WHERE job_id = ?", (job_id_json,)
            )
            if is_completed:
                print("{} job already completed.".format(job_id_json))
            else:
                print("{} job already started.".format(job_id_json))
        return success

    def job_done(self, job_id, results=None):
        if results is None:
            self._execute_sqlite(
                "INSERT OR REPLACE INTO jobs (job_id,is_completed) VALUES(?, 1)",
                (json.dumps(job_id),),
            )
        else:
            self._execute_sqlite(
                "INSERT OR REPLACE INTO jobs (job_id,is_completed,results) VALUES(?, 1,?)",
                (json.dumps(job_id), str(results)),
            )

    def delete_job(self, job_id):
        self._execute_sqlite(
            "DELETE FROM jobs WHERE (job_id = ? )", (json.dumps(job_id),)
        )

    def delete_all_running_jobs(
        self,
    ):
        self._execute_sqlite("DELETE FROM jobs WHERE (is_completed = 0 )")

    def to_pandas(
        self,
    ):
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM jobs", conn)
        except Exception as e:
            print("sqlite error:", str(e))
            df = None
        finally:
            conn.close()
        try:
            df["job_id"] = [json.loads(s) for s in df["job_id"]]
        except:
            pass
        return df
