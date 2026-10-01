// Copyright (c) PODS-AI contributors
// SPDX-License-Identifier: MIT
using CsvHelper.Configuration.Attributes;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace DatasetManager
{
    public class SampleRecord : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        public string Category { get; set; } = "";
        public string NodeName { get; set; } = "";
        public string StartTimestamp { get; set; } = "";
        public string URI { get; set; } = "";
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
