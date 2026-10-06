// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT

using CsvHelper.Configuration.Attributes;
using System.ComponentModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Web;

namespace DatasetManager
{
    public class SampleRecord : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        public string Category { get; set; } = "";
        public string NodeName { get; set; } = "";
        public string StartTimestamp { get; set; } = "";
        public string URI { get; set; } = "";
        public DateTime StartTimestampUtc
        {
            get
            {
                // The URI is in the format "https://live.orcasound.net/bouts/new/port-townsend?time=2026-07-08T18%3A37%3A31.000Z"
                // We can extract the timestamp from the URI and parse it as a DateTime.
                var uri = new Uri(URI);
                var timeValue = HttpUtility.ParseQueryString(uri.Query)["time"];
                if (string.IsNullOrEmpty(timeValue))
                {
                    throw new InvalidOperationException(
                    $"No time parameter found in URI '{URI}'.");
                }

                return DateTime.Parse(
                    timeValue,
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal |
                    DateTimeStyles.AdjustToUniversal);
            }
        }
        public string Description { get; set; } = "";
        public string Notes { get; set; } = "";
        private double? _confidence = null;
        public double? Confidence {
            get => _confidence;
            set
            {
                if (_confidence == value)
                {
                    return;
                }

                _confidence = value;

                OnPropertyChanged();
                OnPropertyChanged(nameof(ConfidenceRatio));
            }
        }
        public double? ConfidenceRatio => Confidence == null ? null : Confidence / 100.0;

        private string _tags = "";
        [Optional]
        public string Tags
        {
            get => _tags;
            set
            {
                if (_tags == value)
                {
                    return;
                }

                _tags = value;

                OnPropertyChanged();
                OnPropertyChanged(nameof(HasTags));
            }
        }
        public bool HasTags => !string.IsNullOrWhiteSpace(Tags);
        protected void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
        public string GetWavFilePath(string baseDirectory)
        {
            string slug = NodeName.Replace('_', '-');
            string path = $"{baseDirectory}\\{Category}\\{slug}_{StartTimestamp}.wav";
            return path;
        }
    }
}
