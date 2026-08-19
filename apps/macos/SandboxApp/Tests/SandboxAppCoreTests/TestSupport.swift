import XCTest

func expect(
    _ condition: @autoclosure () throws -> Bool,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    do {
        XCTAssertTrue(try condition(), file: file, line: line)
    } catch {
        XCTFail("Unexpected error: \(error)", file: file, line: line)
    }
}

func require<T>(
    _ value: @autoclosure () throws -> T?,
    file: StaticString = #filePath,
    line: UInt = #line
) throws -> T {
    try XCTUnwrap(try value(), file: file, line: line)
}

func expectThrows<E: Error & Equatable>(
    _ expected: E,
    file: StaticString = #filePath,
    line: UInt = #line,
    _ body: () throws -> Void
) {
    XCTAssertThrowsError(try body(), file: file, line: line) { error in
        XCTAssertEqual(error as? E, expected, file: file, line: line)
    }
}
