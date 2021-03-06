//
// Created by jagle on 11/20/2019.
//

#include <limits.h>
#include <gtest/gtest.h>

#include <eigen3/Eigen/Dense>

#include "analysis/trace_file_parser.h"

using namespace Eigen;

namespace rlscope {

//TEST(FailingTest, OneEqZero) {
// EXPECT_EQ(1, 0);
//}

class MyEvent {
public:
  std::string _name;
  TimeUsec _start_time_us;
  TimeUsec _duration_us;

  MyEvent() :
      _name(""),
      _start_time_us(0),
      _duration_us(0)
  {
  }

  MyEvent(
      TimeUsec start_time_us,
      TimeUsec end_time_us) :
      _name(""),
      _start_time_us(start_time_us),
      _duration_us(end_time_us - start_time_us)
  {
//    assert(end_time_us >= start_time_us);
  }

  MyEvent(
      const std::string& name,
      TimeUsec start_time_us,
      TimeUsec end_time_us) :
      _name(std::move(name)),
      _start_time_us(start_time_us),
      _duration_us(end_time_us - start_time_us)
  {
//    assert(end_time_us >= start_time_us);
  }

  TimeUsec start_time_us() const {
    return _start_time_us;
  }
  TimeUsec duration_us() const {
    return _duration_us;
  }
  const std::string& name() const {
    return _name;
  }

  bool operator==(const MyEvent& rhs) const {
    auto const& lhs = *this;
    return (lhs._start_time_us == rhs._start_time_us)
           && (lhs._duration_us == rhs._duration_us)
           && (lhs._name == rhs._name);
  }

//  template <typename OStream>
//  friend OStream& operator<<(OStream& os, const MyEvent& obj);

  template <typename OStream>
  friend OStream& operator<<(OStream& os, const MyEvent& obj) {
//    os << "MyEvent(name=" << obj._name;
//    os << ", start=" << obj._start_time_us << " us";
//    os << ", dur=" << obj._duration_us << " us";
//    os << ")";
//    return os;
    obj.Print(os, 0);
    return os;
  }

  template <typename OStream>
  void Print(OStream& os, int indent) const {
    PrintIndent(os, indent);

    os << this->_name << "(";
    os << this->_start_time_us;
    os << ", " << (this->_start_time_us +  this->_duration_us);
    os << ")";

//    os << "MyEvent(name=" << this->_name;
//    os << ", start=" << this->_start_time_us << " us";
//    os << ", dur=" << this->_duration_us << " us";
//    os << ")";

  }

};

//template <>
//std::ostream& operator<<(std::ostream& os, const MyEvent& obj) {
//  os << "MyEvent(name=" << obj._name;
//  os << ", start=" << obj._start_time_us << " us";
//  os << ", dur=" << obj._duration_us << " us";
//  os << ")";
//  return os;
//}

std::ostream& operator<<(std::ostream& os, const MyEvent& obj) {
  obj.Print(os, 0);
  return os;
}


TEST(TestCategoryKey, MapContainsKey) {
  std::map<CategoryKey, int> cmap;
  auto process = "process";
  auto category = "category";
  auto category_key = CategoryKey::FromCategory(process, category);
  cmap[category_key] = 1;
  bool contains = (cmap.find(category_key) != cmap.end());
  EXPECT_EQ(contains, true);
}

TEST(TestCategoryKey, MapContainsKeyCopy) {
  std::map<CategoryKey, int> cmap;
  auto process = "process";
  auto category = "category";
  auto category_key = CategoryKey::FromCategory(process, category);
  cmap[category_key] = 1;
  bool contains = (cmap.find(category_key) != cmap.end());
  EXPECT_EQ(contains, true);


  auto process_copy = "process";
  auto category_copy = "category";
  auto category_key_copy = CategoryKey::FromCategory(process_copy, category_copy);
  contains = (cmap.find(category_key_copy) != cmap.end());
  EXPECT_EQ(contains, true);
}

TEST(TestEigen, ElementWiseLessThan) {
  using IdxArray = Array<size_t, Dynamic, 1>;
  size_t k = 3;
  auto A = IdxArray::Constant(k, 1);
  auto B = IdxArray::Constant(k, 2);
  bool A_less_than_B = (A < B).any();
  EXPECT_TRUE(A_less_than_B);
}

TEST(TestPath, TestGetTraceID) {
  auto const& path = "output/perf_debug/process/ppo2_Walker2DBulletEnv-v0/phase/ppo2_Walker2DBulletEnv-v0/cuda_api_stats.trace_3.proto";
  TraceID trace_id = 0;
  TraceID expect_trace_id = 3;
  MyStatus status = MyStatus::OK();
  status = GetTraceID(path, &trace_id);
  EXPECT_EQ(status.code(), MyStatus::OK().code());
  EXPECT_EQ(trace_id, expect_trace_id);
}

std::vector<MyEvent> InterleaveEvents(const std::map<Operation, std::vector<MyEvent>>& event_map) {
  std::vector<MyEvent> all_events;
  for (const auto &pair : event_map) {
    auto const &op_name = pair.first;
    auto const &events = pair.second;
    for (auto const& event : events) {
      auto event_copy = event;
      event_copy._name = op_name;
      all_events.push_back(event_copy);
    }
  }
  std::sort(all_events.begin(), all_events.end(), [] (const auto& lhs, const auto& rhs) {
    return lhs.start_time_us() < rhs.start_time_us();
  });
  return all_events;
}

TimeUsec EventStartTimeUsec(const std::vector<MyEvent>& events) {
  auto start_time = std::numeric_limits<TimeUsec>::max();
  for (auto const& event : events) {
    start_time = std::min(start_time, event.start_time_us());
  }
  return start_time;
}

std::list<MyEvent> SplitEvents(const std::vector<MyEvent>& op_events) {
  std::list<MyEvent> got_events;
  EventFlattener<MyEvent>::EachOpEvent(
      op_events,
      [&got_events] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
        EXPECT_LE(start_us, end_us);
        got_events.emplace_back(op, start_us, end_us);
      });
  return got_events;
}

TEST(TestProcessing, TestEachOpEvent_01) {
  // A:     [    ]
  // B: [            ]
  //    0   1    2   3
  std::map<Operation, std::vector<MyEvent>> event_map{
      {"A", {{1, 2}}},
      {"B", {{0, 3}}},
  };
  auto const& op_events = InterleaveEvents(event_map);

  auto const& got_events = SplitEvents(op_events);
  std::list<MyEvent> expect_events = {
      {"B", 0, 1},
      {"A", 1, 2},
      {"B", 2, 3},
  };

//  for (auto const& event : op_events) {
//    DBG_LOG("{}", event);
//  }

  EXPECT_EQ(got_events, expect_events);
}

TEST(TestProcessing, TestEachOpEvent_02) {
  // A: [   ]
  // B: [            ]
  //    0   1    2   3
  std::map<Operation, std::vector<MyEvent>> event_map{
      {"A", {{0, 1}}},
      {"B", {{0, 3}}},
  };
  auto const& op_events = InterleaveEvents(event_map);

  auto const& got_events = SplitEvents(op_events);
  std::list<MyEvent> expect_events = {
      {"A", 0, 1},
      {"B", 1, 3},
  };

  EXPECT_EQ(got_events, expect_events);
}

TEST(TestProcessing, TestEachOpEvent_03) {
  // A:          [   ]
  // B: [            ]
  //    0   1    2   3
  std::map<Operation, std::vector<MyEvent>> event_map{
      {"A", {{2, 3}}},
      {"B", {{0, 3}}},
  };
  auto const& op_events = InterleaveEvents(event_map);

  auto const& got_events = SplitEvents(op_events);
  std::list<MyEvent> expect_events = {
      {"B", 0, 2},
      {"A", 2, 3},
  };

  EXPECT_EQ(got_events, expect_events);
}

TEST(TestProcessing, TestEachOpEvent_04) {
  // A:     [   ]   [   ]
  // B: [                   ]
  //    0   1   2   3   4   5
  std::map<Operation, std::vector<MyEvent>> event_map{
      {"A", {{1, 2}, {3, 4}}},
      {"B", {{0, 5}}},
  };
  auto const& op_events = InterleaveEvents(event_map);

  auto const& got_events = SplitEvents(op_events);
  std::list<MyEvent> expect_events = {
      {"B", 0, 1},
      {"A", 1, 2},
      {"B", 2, 3},
      {"A", 3, 4},
      {"B", 4, 5},
  };

  EXPECT_EQ(got_events, expect_events);
}

TEST(TestProcessing, TestEachOpEvent_05) {
  // A:     [   ]   [   ]           [   ]   [   ]
  // B: [                   ]   [                   ]
  //    0   1   2   3   4   5   6   7   8   9   10  11
  std::map<Operation, std::vector<MyEvent>> event_map{
      {"A", {
                {1, 2}, {3, 4},
                {7, 8}, {9, 10}}},
      {"B", {
                {0, 5},
                {6, 11}}},
  };
  auto const& op_events = InterleaveEvents(event_map);

  auto const& got_events = SplitEvents(op_events);
  std::list<MyEvent> expect_events = {

      {"B", 0, 1},
      {"A", 1, 2},
      {"B", 2, 3},
      {"A", 3, 4},
      {"B", 4, 5},

      {"B", 6, 7},
      {"A", 7, 8},
      {"B", 8, 9},
      {"A", 9, 10},
      {"B", 10, 11},

  };

  EXPECT_EQ(got_events, expect_events);
}

TEST(TestProcessing, TestEventFlattener_01) {
  // A:     [   ]   (   )
  // B: [                   ]
  //    0   1   2   3   4   5
  //
  // File 01: [ ]
  // File 02: ( )
  std::map<Operation, std::vector<MyEvent>> event_map_01{
      {"A", {
                {1, 2},
            }},
      {"B", {
                {0, 5},
            }},
  };
  auto const& op_events_file_01 = InterleaveEvents(event_map_01);

  std::map<Operation, std::vector<MyEvent>> event_map_02{
      {"A", {
                {3, 4},
            }},
  };
  auto const& op_events_file_02 = InterleaveEvents(event_map_02);


  std::list<MyEvent> got_events;
  auto append_event = [&got_events] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
    EXPECT_LE(start_us, end_us);
    got_events.emplace_back(op, start_us, end_us);
  };

  EventFlattener<MyEvent> event_flattener;
  TimeUsec expect_file_02_operation_start = 3;
  auto file_02_operation_start = EventStartTimeUsec(op_events_file_02);
  EXPECT_EQ(file_02_operation_start, expect_file_02_operation_start);

  size_t got_total_events = 0;
  size_t expect_total_events = 5;

  event_flattener.ProcessUntil(op_events_file_01, file_02_operation_start, append_event);
  std::list<MyEvent> expect_events_01 = {
      {"B", 0, 1},
  };
  EXPECT_EQ(got_events, expect_events_01);
  got_total_events += got_events.size();
  got_events.clear();

  event_flattener.ProcessUntilFinish(op_events_file_02, append_event);
  std::list<MyEvent> expect_events_02 = {
      {"A", 1, 2},
      {"B", 2, 3},
      {"A", 3, 4},
      {"B", 4, 5},
  };
  EXPECT_EQ(got_events, expect_events_02);
  got_total_events += got_events.size();
  got_events.clear();

  EXPECT_EQ(got_total_events, expect_total_events);

}

//TEST(TestProcessing, TestEventFlattener_02) {
//  // A:     [   ]
//  // B: (           )
//  //    0   1   2   3
//  //
//  // File 01: [ ]
//  // File 02: ( )
//  std::map<Operation, std::vector<MyEvent>> event_map_01{
//      {"A", {
//                {1, 2},
//            }},
//  };
//  auto const& op_events_file_01 = InterleaveEvents(event_map_01);
//
//  std::map<Operation, std::vector<MyEvent>> event_map_02{
//      {"B", {
//                {0, 3},
//            }},
//  };
//  auto const& op_events_file_02 = InterleaveEvents(event_map_02);
//
//
//  std::list<MyEvent> got_events;
//  auto append_event = [&got_events] (const Operation& op, TimeUsec start_us, TimeUsec end_us) {
//    EXPECT_LE(start_us, end_us);
//    got_events.emplace_back(op, start_us, end_us);
//  };
//
//  auto file_02_operation_start = EventStartTimeUsec(op_events_file_02);
//  EXPECT_EQ(file_02_operation_start, 0);
//
//  size_t got_total_events = 0;
//  size_t expect_total_events = 3;
//
//  EventFlattener<MyEvent> event_flattener;
//  event_flattener.ProcessUntil(op_events_file_01, file_02_operation_start, append_event);
//  std::list<MyEvent> expect_events_01 = {
//  };
//  EXPECT_EQ(got_events, expect_events_01);
//  got_total_events += got_events.size();
//  got_events.clear();
//
//  event_flattener.ProcessUntilFinish(op_events_file_02, append_event);
//  std::list<MyEvent> expect_events_02 = {
//      {"B", 0, 1},
//      {"A", 1, 2},
//      {"B", 2, 3},
//  };
//  EXPECT_EQ(got_events, expect_events_02);
//  got_total_events += got_events.size();
//  got_events.clear();
//
//  EXPECT_EQ(got_total_events, expect_total_events);
//
//}

TEST(TestUtil, TestKeepExtreme_01) {
  auto got = KeepExtreme(
      std::vector<TimeUsec>{0, 5, 3, 4},
      0,
      /*should_keep=*/[] (TimeUsec x, TimeUsec min_x) {
        return x < min_x;
      },
      /*should_skip=*/[] (TimeUsec x) {
        return x == 0;
      });
  auto expect = 3;
  EXPECT_EQ(got, expect);
}

TEST(TestUtil, TestKeepExtreme_02) {
  //[2019-12-18 19:37:25.488] [debug] [trace_file_parser.h:2590] pid=21880 @ MergeInplace: region1 before: RegionMetadataReducer(start=0 us, end=0 us, num_events=0)
  //[2019-12-18 19:37:25.488] [debug] [trace_file_parser.h:2591] pid=21880 @ MergeInplace: region2 before: RegionMetadataReducer(start=1571625977177087000 us, end=1571625998937912000 us, num_events=140)
  //[2019-12-18 19:37:25.488] [debug] [trace_file_parser.h:2615] pid=21880 @ MergeInplace: region1 after: RegionMetadataReducer(start=-647324648 us, end=-361336128 us, num_events=140)

  TimeUsec got_start = KeepExtreme<TimeUsec>(
      std::vector<TimeUsec>{0, 1571625977177087000},
      0,
      /*should_keep=*/[] (TimeUsec x, TimeUsec min_x) {
        return x < min_x;
      },
      /*should_skip=*/[] (TimeUsec x) {
        return x == 0;
      });
  auto expect_start = 1571625977177087000;
  EXPECT_EQ(got_start, expect_start);

  TimeUsec got_end = KeepExtreme<TimeUsec>(
      std::vector<TimeUsec>{0, 1571625998937912000},
      0,
      /*should_keep=*/[] (TimeUsec x, TimeUsec min_x) {
        return x < min_x;
      },
      /*should_skip=*/[] (TimeUsec x) {
        return x == 0;
      });
  auto expect_end = 1571625998937912000;
  EXPECT_EQ(got_end, expect_end);

}

TEST(TestEigen, ArrayModifyOne) {
//  using IdxArray = Array<size_t, Dynamic, 1>;
//  size_t k = 3;
//  auto A = IdxArray::Constant(k, 0);
//  IdxArray B;
//  B << 0, 1, 0;
//  // A(1) += 1;
//  A[1][0] = 1;
//  bool A_eq_B = (A == B).all();
//  EXPECT_TRUE(A_eq_B);

//  // using ArrayType = ArrayXXf;
//  using ArrayType = Array<size_t, Dynamic, Dynamic>;
//
//  ArrayType  m(2,2);
//
//  // assign some values coefficient by coefficient
//  m(0,0) = 1.0; m(0,1) = 2.0;
//  m(1,0) = 3.0; m(1,1) = m(0,1) + m(1,0);
//
//  ArrayType  m_comma(2,2);
//    // using the comma-initializer is also allowed
//  m_comma <<
//      1.0, 2.0,
//      3.0, 5.0;
//  EXPECT_TRUE((m == m_comma).all());


//  // using ArrayType = ArrayXXf;
//  using ArrayType = Array<size_t, Dynamic, 1>;
//
//  ArrayType  m(2,1);
//
//  // assign some values coefficient by coefficient
//  m(0,0) = 1.0;
//  m(1,0) = 3.0;
//
//  ArrayType  m_comma(2,1);
//  // using the comma-initializer is also allowed
//  m_comma <<
//      1.0,
//      3.0;
//  EXPECT_TRUE((m == m_comma).all());


  // using ArrayType = ArrayXXf;
  using ArrayType = Array<size_t, Dynamic, 1>;

  ArrayType  m(2);

  // assign some values coefficient by coefficient
  m(0) = 1.0;
  m(1) = 3.0;
  m(1) += 3.0;

  ArrayType  m_comma(2);
  // using the comma-initializer is also allowed
  m_comma <<
      1.0,
      6.0;
  EXPECT_TRUE((m == m_comma).all());

}

}
