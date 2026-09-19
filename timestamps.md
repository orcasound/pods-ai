# Timestamps

Timestamps are complicated.  This document summarizes the use of timestamps across relevant
Orcasound systems.

## AWS S3

In AWS S3, the name of each folder holding audio clips indicates the time (in Unix seconds since 1970,
where [EpochConverter](https://www.epochconverter.com/) is a handy online site for conversion)
at which the container on the RPI restarted.  Audio then begins to stream about 2-3 seconds later
(hence AUDIO_OFFSET_SECONDS=2 in some scripts), using files named `liveN.ts` where `N` is an integer
counting up from 0.  Each ts file has a ~10 second segment.  It isn't exactly 10 seconds, but the
drift across the course of a day is less than a second, so as long as the RPI container restarts every
day then 10 seconds is a useful approximation. 

RPIs running new code will write the local time of each 10-second audio clip into the HLS manifest using
`#EXT-X-PROGRAM-DATE-TIME`, to not have to compute the time and deal with the 2-3 second correction:

| feed          | last audio before update | updated    |
| ------------- | ------------------------ | ---------- |
| orcasound-lab	| 2026-05-28T04:29:02Z     | 2026-05-28 |
| sunset-bay    | 2026-07-04T18:03:34Z     | 2026-08-04 |

## Orcasite

At the time of this writing, the Orcasite bouts interface does not have the 2-3 second correction and hence
shows a time 2-3 seconds earlier than the actual clock time of the recording.  This discrepancy can be heard in
[Dave Thaler's timing test](https://live.orcasound.net/bouts/new/andrews-bay?time=2025-10-17T23:39:59.000Z).
Similarly, the Orcasite URI to view a bout must have the incorrect timestamp in order to hear the correct
audio, so computing the Orcasite URI from actual time requires subtracting 2 seconds from the real clock time
and embedding the result in the URI.  Fixing Orcasite to use the correct time is tracked by
[Orcasite issue 1041](https://github.com/orcasound/orcasite/issues/1041).

Note that having Orcasite use [#EXT-X-PROGRAM-DATE-TIME](https://github.com/orcasound/orcasite/issues/1035)
when it exists would avoid the discrepancy for such audio, but if the discrepancy is not fixed at the same
time, then the bouts interface will be inconsistent in that it would show the correct time for some audio
and the incorrect time for other audio.

Currently the URIs in the csv files in this repository go to the Orcasite bouts interface and so
are intentionally off by 2 seconds.  That is, the URIs do not use the detection time, they show "Orcasite" time.
And the Timestamp column of `training_3s_samples.csv` is similarly the Orcasite start time, not the correct time.
In this repository, `download_audio_segment()` in `download_wavs.py` thus uses the Orcasite time to find the
audio to download.  If the csv file is changed to use real time, this will need to change.

## OrcaHello database

In the OrcaHello database, each detection has a `timestamp` field that is the _start_ time of a
60-second detection.  Prior to 2025-10-12T14:23:00Z (the start of the current "epoch"),
OrcaHello had bugs where the timestamp in the database is incorrect, and the longer since the
folder timestamp, the more the timestamp would be off.  Fortunately, the algorithm to correct
the timestamp is known and so the [FixTimestampsAsync](https://github.com/orcasound/orcahello/blob/main/NotificationSystem/NotificationSystem/Models/OrcasiteHelper.cs#L579) method in the OrcaHello repository can
compute the corrected timestamp.
